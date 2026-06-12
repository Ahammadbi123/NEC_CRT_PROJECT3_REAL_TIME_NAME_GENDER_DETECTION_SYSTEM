# Real-Time Name and Gender Detection System using CNN and OpenCV

An AI-powered system built with Python, OpenCV, and TensorFlow to identify registered users and predict their gender from live webcam feeds using Convolutional Neural Networks (CNN).

## 🎯 Project Overview
This project demonstrates a full machine learning pipeline including:
*   **Data Collection:** Automated face capture for user registration.
*   **Model Training:** Training a custom CNN model on collected datasets.
*   **Live Inference:** Real-time face recognition and gender classification.

## 🚀 Features
*   ✅ **User Registration:** Captures 30 face images per user.
*   ✅ **Deep Learning Model:** 3-layer CNN architecture for high accuracy.
*   ✅ **Gender Detection:** Real-time gender display (Male/Female).
*   ✅ **Unknown Recognition:** Identifies unregistered faces as "Unknown".
*   ✅ **Real-time Performance:** Low-latency detection using OpenCV.

## 📁 Project Structure
```text
face_gender_project/
├── dataset/            # Captured face images
├── models/             # Trained .h5 model and labels
├── database/           # User info in JSON format
├── registration/       # Module for adding new users
├── training/           # CNN model training logic
├── detection/          # Real-time inference code
├── main.py             # Main entry point
└── requirements.txt    # Project dependencies
```

## 🛠️ Tech Stack
*   **Language:** Python 3.10+
*   **Computer Vision:** OpenCV
*   **Deep Learning:** TensorFlow, Keras
*   **Data Handling:** NumPy, JSON
*   **Image Processing:** Pillow

## 💻 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Ahammadbi123/NEC_CRT_PROJECT3_REAL_TIME_NAME_GENDER_DETECTION_SYSTEM.git
cd NEC_CRT_PROJECT3_REAL_TIME_NAME_GENDER_DETECTION_SYSTEM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```

## 📝 Workflow
1.  **Register:** Select Option 1 to add a new user and capture photos.
2.  **Train:** Select Option 2 to train the CNN model on the new dataset.
3.  **Detect:** Select Option 3 to start live webcam detection.

## 🎓 Model Architecture
*   **Input Layer:** 64x64 RGB Images
*   **Conv Blocks:** 3 Convolutional Layers with MaxPooling and BatchNormalization.
*   **Dense Layers:** Fully connected layers with Dropout (0.5) to prevent overfitting.
*   **Output:** Softmax activation for multi-class classification.

## 🤝 Support
For any issues or questions, feel free to open an issue or reach out to the author.

---
**Developed By:** SHAIK AHAMMAD BI
**Professional Passionate about Machine Learning and AI solutions.**
**Updated:** June 2026
