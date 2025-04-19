import cv2
import json
from pathlib import Path
from config import MODEL_FILE, LABELS_FILE, CONFIDENCE_THRESHOLD

class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.id_to_components = {}
        
        self.load_model()
        self.load_labels()
    
    def load_model(self):
        if not Path(MODEL_FILE).exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")
        self.recognizer.read(MODEL_FILE)
    
    def load_labels(self):
        if not Path(LABELS_FILE).exists():
            raise FileNotFoundError(f"Label file not found at {LABELS_FILE}")
        
        with open(LABELS_FILE, 'r') as f:
            label_data = json.load(f)
        
        for label_str, label_id in label_data.items():
            parts = label_str.split('_')
            if len(parts) == 3:  # Format: ethnicity_name_emotion
                self.id_to_components[label_id] = {
                    'ethnicity': parts[0],
                    'name': parts[1],
                    'emotion': parts[2]
                }
    
    def recognize_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        recognition_results = []
        
        for (x, y, w, h) in faces:
            label_id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
            
            if confidence < CONFIDENCE_THRESHOLD and label_id in self.id_to_components:
                components = self.id_to_components[label_id]
                recognition_results.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "name": components['name'],
                    "emotion": components['emotion'],
                    "ethnicity": components['ethnicity'],
                    "confidence": float(confidence),
                    "recognized": True
                })
            else:
                recognition_results.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "confidence": float(confidence),
                    "recognized": False
                })
        
        return recognition_results