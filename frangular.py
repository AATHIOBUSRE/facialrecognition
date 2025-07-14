import cv2 as cv
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy import dot
from numpy.linalg import norm
import math

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

# Predict function
def predict(test_image_path):
    # Load models and embeddings
    model, encoder, embeddings, labels = load_models_and_embeddings()

    # Read and preprocess the test image
    test_img = cv.imread(test_image_path)
    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2RGB)
    x, y, w, h = detector.detect_faces(test_img)[0]['box']
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
    
    # Print results
    if 1-confidence_score > 0.7:
        print(f'Predicted: {predicted_label}, Confidence: {1 - confidence_score}')
    else:
        print("Unknown")
# Test the function with an image
predict('images\\Bill Gates66_600.jpg')
