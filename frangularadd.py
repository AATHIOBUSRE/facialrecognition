import cv2 as cv
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy import dot
from numpy.linalg import norm
import math
import os

# Initialize MTCNN for face detection and FaceNet for embeddings
detector = MTCNN()
embedder = FaceNet()

# Function to get embeddings
def get_embedding(face_img):
    face_img = face_img.astype('float32')  # Convert image to float
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # Return 512D embedding

# Angular distance function
def angular_distance(embedding1, embedding2):
    cos_sim = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angle = math.acos(cos_sim) / math.pi
    return angle

# Predict with angular distance
def predict_with_angular_distance(test_embedding, embeddings, labels, threshold=0.7):
    best_score = float('inf')
    best_label = None

    for i, embedding in enumerate(embeddings):
        ang_dist = angular_distance(test_embedding, embedding)

        if ang_dist < best_score:
            best_score = ang_dist
            best_label = labels[i]

    if best_score < threshold:
        return best_label, best_score
    else:
        return "Unknown", best_score

# Load the saved models and encoder
def load_models_and_embeddings():
    with open('model\\model_svm_few.pickle', 'rb') as f:
        model = pickle.load(f)

    with open('model\\label_encoder_few.pickle', 'rb') as f:
        encoder = pickle.load(f)

    npzfile = np.load('model\\faces_embeddings_few1.npz')
    embeddings = npzfile['arr_0']
    labels = npzfile['arr_1']

    return model, encoder, embeddings, labels

# Save updated embeddings and labels
def save_embeddings_and_labels(embeddings, labels):
    np.savez_compressed('model\\faces_embeddings_few1.npz', embeddings, labels)

# Add new images and generate embeddings
def add_new_images(paths, person_name):
    new_embeddings = []
    new_labels = []

    for path in paths:
        if os.path.isdir(path):
            # Process images in a directory
            for im_name in os.listdir(path):
                img_path = os.path.join(path, im_name)
                if os.path.isfile(img_path):
                    img = cv.imread(img_path)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    results = detector.detect_faces(img)
                    if results:
                        x, y, w, h = results[0]['box']
                        x, y = abs(x), abs(y)
                        face = img[y: y+h, x: x+w]
                        face = cv.resize(face, (160, 160))
                        embedding = get_embedding(face)
                        new_embeddings.append(embedding)
                        new_labels.append(person_name)
        elif os.path.isfile(path):
            # Process individual image files
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = detector.detect_faces(img)
            if results:
                x, y, w, h = results[0]['box']
                x, y = abs(x), abs(y)
                face = img[y: y+h, x: x+w]
                face = cv.resize(face, (160, 160))
                embedding = get_embedding(face)
                new_embeddings.append(embedding)
                new_labels.append(person_name)
        else:
            print(f"Path {path} does not exist or is not a file/directory.")
    
    if len(new_embeddings) == 0:
        print("No valid images found in the provided paths.")
        return

    new_embeddings = np.array(new_embeddings)
    new_labels = np.array(new_labels)

    # Load current embeddings and labels
    model, encoder, old_embeddings, old_labels = load_models_and_embeddings()

    # Append new data to the existing data
    updated_embeddings = np.concatenate((old_embeddings, new_embeddings), axis=0)
    updated_labels = np.concatenate((old_labels, new_labels), axis=0)

    # Save the updated embeddings and labels
    save_embeddings_and_labels(updated_embeddings, updated_labels)

    print(f"Added {len(new_labels)} new images for {person_name}.")

# Predict function
def predict(test_image_path):
    # Load models and embeddings
    model, encoder, embeddings, labels = load_models_and_embeddings()

    # Read and preprocess the test image
    test_img = cv.imread(test_image_path)
    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2RGB)
    results = detector.detect_faces(test_img)
    if not results:
        print("No faces detected")
        return

    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)
    test_image = test_img[y: y+h, x: x+w]
    test_image = cv.resize(test_image, (160, 160))
    test_embedding = get_embedding(test_image)

    # Predict using the SVM model
    predicted_label_encoded = model.predict([test_embedding])[0]

    # Check if the predicted label is in the encoder's classes
    if predicted_label_encoded in encoder.classes_:
        predicted_label = encoder.inverse_transform([predicted_label_encoded])[0]
    else:
        predicted_label = "Unknown"

    # Optionally use angular distance for additional verification
    predicted_label, confidence_score = predict_with_angular_distance(test_embedding, embeddings, labels, threshold=0.7)
    
    # Print results and prompt user to add images
    if 1 - confidence_score > 0.7:
        print(f'Predicted: {predicted_label}, Confidence: {1 - confidence_score}')
    else:
        print("Unknown")
   
    # Ask user if they want to add more images
    response = input("add more images? (yes/no): ").strip().lower()
    if response == 'yes':
        num_paths = int(input("How many paths do you want to provide? "))
        paths = []
        for i in range(num_paths):
            path = input(f"Enter the path to file or directory {i+1}: ").strip()
            paths.append(path)
        
        # Ask user for the label for new images
        person_name = input("Enter the name of the person for whom you want to add images: ").strip()
        
        add_new_images(paths, person_name)

# Test the function with an image
predict('105_classes_pins_dataset\pins_Brian J. Smith\Brian J. Smith0_619.jpg')