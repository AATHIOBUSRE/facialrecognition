import cv2 as cv
import numpy as np
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
import pickle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
class FaceRecognitionPipeline:
    def __init__(self, model_path, encoder_path):
        # Load pre-trained model and encoder
        self.model = pickle.load(open(model_path, 'rb'))
        self.encoder = pickle.load(open(encoder_path, 'rb'))
        self.detector = MTCNN()
        self.embedder = FaceNet()

    def extract_face(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = self.detector.detect_faces(img)
        if not result:
            raise ValueError("No face detected")
        x, y, w, h = result[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face = cv.resize(face, (160, 160))
        return face

    def get_embedding(self, face_img):
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        yhat = self.embedder.embeddings(face_img)
        return yhat[0]

    def predict(self, img):
        try:
            face_img = self.extract_face(img)
            face_embedding = self.get_embedding(face_img)
            face_embedding = np.expand_dims(face_embedding, axis=0)
            probabilities = self.model.predict_proba(face_embedding)[0]
            prediction = self.model.predict(face_embedding)[0]
            predicted_class = self.encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]

            if confidence < 0.40:
                return "Unknown", confidence
            else:
                return predicted_class, confidence
        except ValueError:
            return "No face detected", None

# Initialize the pipeline
pipeline = FaceRecognitionPipeline('model/model_svm.pickle', 'model/label_encoder.pickle')

# Example usage for prediction
test_img = cv.imread('images\adriana.jpg')

predicted_class, confidence = pipeline.predict(test_img)
print(f'Predicted class: {predicted_class}')
print(f'Confidence: {confidence:.4f}')
