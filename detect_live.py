import cv2
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from utils import get_face_classifier

def detect():
    # మోడల్ ఉందో లేదో చెక్ చేయడం
    model_path = 'models/face_model.h5'
    if not os.path.exists(model_path):
        print("Error: Model file not found. Please train the model first (Option 2).")
        return

    model = load_model(model_path)
    label_map = np.load('models/label_map.npy', allow_pickle=True).item()
    inv_label_map = {v: k for k, v in label_map.items()}
    
    with open('database/users.json', 'r') as f:
        user_db = json.load(f)
    
    face_cascade = get_face_classifier()
    cam = cv2.VideoCapture(0)

    print("Live Detection Started. Press 'q' to exit.")

    while True:
        ret, frame = cam.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (64, 64)) / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            
            
            prediction = model.predict(face_img, verbose=0)
            idx = np.argmax(prediction)
            conf = np.max(prediction)
            
            
            if conf > 0.8:
                name = inv_label_map[idx]
                gender = user_db[name]['gender']
                
                label = f"{name} ({gender}) {int(conf*100)}%"
                color = (0, 255, 0) # Green box
            else:
                label = f"Unknown ({int(conf*100)}%)"
                color = (0, 0, 255) # Red box

            
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Live Detection", frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()