import cv2
import os

def get_face_classifier():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_dirs():
    dirs = ['dataset', 'models', 'database']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)