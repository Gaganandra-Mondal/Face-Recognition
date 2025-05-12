import cv2
import os
from deepface import DeepFace

db_path = "face_db"
model_name = "Facenet512"         
detection_backend = "opencv"     

os.makedirs(db_path, exist_ok=True)


def capture_face_image(name="with you"):
    save_path = os.path.join(db_path, f"{name}.jpg")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        exit()

    print("Press 's' to save your face or 'q' to quit.")
    i = 1
    while i > 0:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not captured.")
            break

        cv2.imshow("Capture Your Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(save_path, frame)
            print(f"Saved your face as {save_path}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def recognize_face():
    name = "with you"  

  
    captured_image_path = os.path.join(db_path, f"{name}.jpg")
    if not os.path.exists(captured_image_path):
        print("No face image found in the database. Capturing your face...")
        capture_face_image(name)

  
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        exit()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        try:
            
            result = DeepFace.verify(frame, captured_image_path, model_name=model_name, detector_backend=detection_backend)

            if result['verified']:
                text = f"Face matched: {name}"
                color = (0, 255, 0)  
            else:
                text = "No match found"
                color = (0, 0, 255)  

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            text = "Detection error"
            color = (0, 0, 255)  

       
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


recognize_face()
