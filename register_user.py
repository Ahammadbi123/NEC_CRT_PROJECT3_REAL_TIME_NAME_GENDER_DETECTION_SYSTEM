import cv2
import os
import json
from utils import get_face_classifier

def register():
    name = input("Enter Name: ")
    gender = input("Enter Gender (Male/Female): ")
    
    save_path = f"dataset/{name}"
    os.makedirs(save_path, exist_ok=True)
    
    cam = cv2.VideoCapture(0)
    face_cascade = get_face_classifier()
    count = 0
    
    print("Capturing 30 images... Press SPACE to capture.")
    
    while count < 30:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                count += 1
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
                print(f"Captured {count}/30")
        
        cv2.imshow("Registering", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cam.release()
    cv2.destroyAllWindows()

    # Update database
    db_path = 'database/users.json'
    with open(db_path, 'r') as f:
        data = json.load(f)
    data[name] = {"gender": gender}
    with open(db_path, 'w') as f:
        json.dump(data, f, indent=4)
    print("User Registered Successfully!")

if __name__ == "__main__":
    register()