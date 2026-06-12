import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2

def train():
    X, y = [], []
    labels = os.listdir('dataset')
    label_map = {name: i for i, name in enumerate(labels)}
    
    for label in labels:
        path = f'dataset/{label}'
        for img_name in os.listdir(path):
            img = cv2.imread(f"{path}/{img_name}")
            img = cv2.resize(img, (64, 64))
            X.append(img)
            y.append(label_map[label])
    
    X = np.array(X) / 255.0
    y = np.array(y)
    
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(labels), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10)
    
    model.save('models/face_model.h5')
    np.save('models/label_map.npy', label_map)
    print("Model Trained and Saved!")

if __name__ == "__main__":
    train()