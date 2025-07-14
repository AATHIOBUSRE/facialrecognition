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
import random
import pickle
from numpy import dot
from numpy.linalg import norm

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load and visualize an example image
img = cv.imread('images\\Bill Gates66_600.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)

# Initialize MTCNN for face detection
detector = MTCNN()
results = detector.detect_faces(img)
x, y, w, h = results[0]['box']
img = cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
plt.imshow(img)

# Crop face
my_face = img[y:y+h, x:x+w]
my_face = cv.resize(my_face, (160, 160))
plt.imshow(my_face)

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
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory + '/' + sub_dir + '/'
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

# Initialize FaceNet embedder
embedder = FaceNet()

# Get embeddings
def get_embedding(face_img):
    face_img = face_img.astype('float32')  # Convert image to float
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # Return 512D embedding

# Create embeddings for the dataset
EMBEDDED_X = [get_embedding(img) for img in X]
EMBEDDED_X = np.asarray(EMBEDDED_X)
np.savez_compressed('faces_embeddings.npz', EMBEDDED_X, Y)

# Encode labels
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

# Save the encoder
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=42)

# Train an SVM classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# Save the model
pickle.dump(model, open('model_svm.pickle', mode='wb'))

# Cosine similarity function
def cosine_similarity(embedding1, embedding2):
    cos_sim = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    return cos_sim

# Predict with cosine similarity
def predict_with_cosine_similarity(test_embedding, embeddings, labels, threshold=0.7):
    best_score = -1
    best_label = None
    
    for i, embedding in enumerate(embeddings):
        sim_score = cosine_similarity(test_embedding, embedding)
        
        if sim_score > best_score:
            best_score = sim_score
            best_label = labels[i]
    
    # If the highest cosine similarity score exceeds the threshold, consider it a confident prediction
    if best_score > threshold:
        return best_label, best_score
    else:
        return "Unknown", best_score

# Test image
test_img = cv.imread('newdata\\val\\alycia\\alycia dabnem carey7_185.jpg')
test_img = cv.cvtColor(test_img, cv.COLOR_BGR2RGB)
x, y, w, h = detector.detect_faces(test_img)[0]['box']
x, y = abs(x), abs(y)
test_image = test_img[y: y+h, x: x+w]
test_image = cv.resize(test_image, (160, 160))
test_embedding = get_embedding(test_image)

# Predict the label and confidence score
predicted_label, confidence_score = predict_with_cosine_similarity(test_embedding, EMBEDDED_X, Y, threshold=0.7)
predicted_label = encoder.inverse_transform([predicted_label])[0] if predicted_label != "Unknown" else "Unknown"

print(f'Predicted: {predicted_label}, Confidence: {confidence_score}')
print('Expected: alycia')

# Plot the test image
plt.imshow(test_img)
plt.show()
