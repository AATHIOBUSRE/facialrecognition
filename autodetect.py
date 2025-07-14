import cv2 as cv
import numpy as np
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

class FaceRecognitionPipeline:
    def __init__(self, model_path, encoder_path, embeddings_path):
        # Load pre-trained model, encoder, and dataset
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.embeddings_path = embeddings_path
        
        self.model = pickle.load(open(self.model_path, 'rb'))
        self.encoder = pickle.load(open(self.encoder_path, 'rb'))
        self.detector = MTCNN()
        self.embedder = FaceNet()

        # Load embeddings and labels from .npz file
        data = np.load(self.embeddings_path)
        self.embeddings = data['embeddings'].tolist()
        self.labels = data['labels'].tolist()

        # Initialize augmentation
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

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

    def augment_image(self, image):
        image = np.expand_dims(image, 0)  # Convert to 4D array for augmentation
        augmented_images = []

        it = self.datagen.flow(image, batch_size=1)
        for _ in range(20):  # Generate 20 augmented images
            batch = next(it)  # Use next() to get the next batch of augmented images
            augmented_image = batch[0].astype('uint8')
            augmented_images.append(augmented_image)
        
        return augmented_images

    def add_new_image(self, img, label):
        try:
            # Extract face and get embedding
            face_img = self.extract_face(img)
            face_embedding = self.get_embedding(face_img)

            # Add the new embedding and label to the dataset
            self.embeddings.append(face_embedding)
            self.labels.append(label)

            # Augment the image and add embeddings
            augmented_images = self.augment_image(face_img)
            for aug_img in augmented_images:
                aug_embedding = self.get_embedding(aug_img)
                self.embeddings.append(aug_embedding)
                self.labels.append(label)

            # Update the encoder with the new label if necessary
            if label not in self.encoder.classes_:
                self.encoder.classes_ = np.append(self.encoder.classes_, label)

            # Encode the labels
            encoded_labels = self.encoder.transform(self.labels)

            # Retrain the SVM model with the updated dataset
            self.model = SVC(kernel='linear', probability=True)
            self.model.fit(self.embeddings, encoded_labels)

            # Save the updated model, encoder, and embeddings
            with open(self.model_path, 'wb') as model_file:
                pickle.dump(self.model, model_file)
            with open(self.encoder_path, 'wb') as encoder_file:
                pickle.dump(self.encoder, encoder_file)
            np.savez_compressed(self.embeddings_path, embeddings=np.asarray(self.embeddings), labels=np.asarray(self.labels))

            print(f"Added new image with label: {label}")
        except ValueError:
            print("No face detected, unable to add new image.")

    def handle_unknown(self, img):
        # Ask the user if they want to add the unknown face
        add_new = input("Face not recognized. Do you want to add this face to the database? (yes/no): ").strip().lower()
        if add_new == 'yes':
            # Get the person's name
            person_name = input("Enter the name of the person: ").strip()

            # Add the new image to the model
            self.add_new_image(img, person_name)
            print(f"New face added with name: {person_name}")
        else:
            print("Face not added.")

    def predict(self, img):
        try:
            face_img = self.extract_face(img)
            face_embedding = self.get_embedding(face_img)
            face_embedding = np.expand_dims(face_embedding, axis=0)
            probabilities = self.model.predict_proba(face_embedding)[0]
            prediction = self.model.predict(face_embedding)[0]
            predicted_class = self.encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]

            if confidence < 0.60:
                return "Unknown", confidence
            else:
                return predicted_class, confidence
        except ValueError:
            return "No face detected", None

# Example usage
pipeline = FaceRecognitionPipeline('model\\model_svm.pickle', 'model\\label_encoder.pickle', 'model\\faces_embeddings_named.npz')

# Test prediction
test_img = cv.imread('images\\Cristiano Ronaldo42_1352.jpg')
predicted_class, confidence = pipeline.predict(test_img)
print(f'Predicted class: {predicted_class}')
print(f'Confidence: {confidence:.4f}')

# Handle unknown faces
if predicted_class == "Unknown":
    pipeline.handle_unknown(test_img)