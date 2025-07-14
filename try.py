import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from numpy import dot
from numpy.linalg import norm
import math
import random

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MTCNN for face detection
detector = MTCNN()

# Function to get embeddings
def get_embedding(face_img):
    face_img = face_img.astype('float32')  # Convert image to float
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    embedder = FaceNet()
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # Return 512D embedding

# Class for loading faces
class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.size)
        return face_arr

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                print(e)
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)

# Load face images and labels
faceloading = FACELOADING("data\\train")
X, Y = faceloading.load_classes()

# Function to visualize faces
def plot_faces(num_images=20):
    if len(X) < num_images:
        print(f"Only {len(X)} faces available to plot.")
        num_images = len(X)
    selected_indices = random.sample(range(len(X)), num_images)
    selected_faces = [X[i] for i in selected_indices]
    plt.figure(figsize=(10, 10))
    for i, face in enumerate(selected_faces):
        plt.subplot(4, 5, i + 1)
        plt.imshow(face)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_faces()

# Create embeddings for the dataset
EMBEDDED_X = [get_embedding(img) for img in X]
EMBEDDED_X = np.asarray(EMBEDDED_X)

# Save embeddings and labels
np.savez_compressed('model\\faces_embeddings_few1.npz', EMBEDDED_X, Y)

# Encode labels
encoder = LabelEncoder()
encoder.fit(Y)
Y_encoded = encoder.transform(Y)

# Save the encoder
with open('model\\label_encoder_few.pickle', 'wb') as f:
    pickle.dump(encoder, f)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y_encoded, shuffle=True, random_state=42)

# Train an SVM classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# Save the model
with open('model\\model_svm_few.pickle', 'wb') as f:
    pickle.dump(model, f)

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

# Test image prediction
def predict(test_image_path):
    test_img = cv.imread(test_image_path)
    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2RGB)
    x, y, w, h = detector.detect_faces(test_img)[0]['box']
    x, y = abs(x), abs(y)
    test_image = test_img[y: y+h, x: x+w]
    test_image = cv.resize(test_image, (160, 160))
    test_embedding = get_embedding(test_image)

    # Predict using SVM model
    with open('model\\model_svm_few.pickle', 'rb') as f:
        model = pickle.load(f)
    
    # Predict label using SVM
    predicted_label_encoded = model.predict([test_embedding])[0]
    predicted_label = encoder.inverse_transform([predicted_label_encoded])[0] if predicted_label_encoded != "Unknown" else "Unknown"

    # Optionally use angular distance
    predicted_label, confidence_score = predict_with_angular_distance(test_embedding, EMBEDDED_X, Y_encoded, threshold=0.7)
    predicted_label = encoder.inverse_transform([predicted_label])[0] if predicted_label != "Unknown" else "Unknown"

    print(f'Predicted: {predicted_label}, Confidence: {1 - confidence_score}')
    
    # Plot the test image
    plt.imshow(test_img)
    plt.show()

# Test the function with an image
predict('images\\adriana.jpg')
