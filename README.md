# Facial Recognition Web App 👤📸

A complete Flask-based web application for real-time facial recognition using MTCNN for face detection, FaceNet for embeddings, and SVM for classification. The UI is built using HTML5 and Bootstrap 5 with camera support.

## 🔧 Features

- Upload image to **predict face**
- Upload image with label to **add new person**
- Use **camera** to capture face for prediction or addition
- Styled with modern **Bootstrap cards and modals**
- Works entirely in browser and Flask backend
- Supports face augmentation and dynamic retraining on-the-fly


## 🚀 Installation & Setup

1. **Clone the repository**


git clone https://github.com/your-username/facial-recognition-app.git
cd facial-recognition-app
Create a Conda environment (Python 3.10 recommended)


conda create -n facerecog python=3.10
conda activate facerecog
Install dependencies


pip install -r requirements.txt
Your requirements.txt should include:


Flask
opencv-python
mtcnn
keras-facenet
scikit-learn
numpy
tensorflow==2.13  # ensure Python version compatibility
⚠️ If you're using Python 3.12, downgrade to 3.10 as some packages may not be supported.

Run the Flask app

python app.py
Open the app in your browser

http://127.0.0.1:5000
🧠 How it Works
MTCNN detects faces in uploaded or captured images.

FaceNet extracts 512D embeddings for each face.

Embeddings are passed to an SVM model to classify.

Unknown faces can be added dynamically with augmentation.

Model retrains and persists automatically on addition.

📸 Camera Access
The web app allows:

Capturing photos directly from webcam

Using the captured photo for prediction or adding new faces

Works in Chrome, Edge, Firefox (permissions required)

🛑 Mobile support is limited depending on browser capabilities.

🧪 Example Usage
Click Predict Image → Upload or capture → See prediction and confidence.

Click Add Image → Upload with name or use camera → Automatically augments and retrains the model.

✅ Tips

Use good lighting and front-facing poses.

Monitor model/ folder for updated models and embeddings.

📃 License
MIT License

🙌 Acknowledgments
FaceNet

MTCNN

Bootstrap 5 UI framework

Built with 💡 by Aathi Obusre M

