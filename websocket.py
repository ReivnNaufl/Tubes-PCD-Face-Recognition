import cv2
import asyncio
import joblib
import numpy as np
from fastapi import WebSocket
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
from logic.LBPH_Face_Recognition import FaceRecognizer as LBPHRecognizer
from config import FACENET_MODEL_FILE, CONFIDENCE_THRESHOLD

class WebSocketManager:
    def __init__(self):
        self.active_connections = {}
        self.camera_task = None
        self.camera_active = False
        self.current_model = None
        
        # Initialize models lazily
        self.lbph_recognizer = None
        self.facenet_recognizer = None
        self.face_cascade = None

    async def initialize_models(self, model_type):
        """Lazy initialization of models"""
        if self.face_cascade is None:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        if model_type == 'lbph' and self.lbph_recognizer is None:
            self.lbph_recognizer = LBPHRecognizer()
        elif model_type == 'facenet' and self.facenet_recognizer is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.facenet_recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            facenet_data = joblib.load(FACENET_MODEL_FILE)
            self.facenet_knn = facenet_data['model']

    async def connect(self, websocket: WebSocket, model_type: str):
        await websocket.accept()
        if model_type not in self.active_connections:
            self.active_connections[model_type] = []
        self.active_connections[model_type].append(websocket)
        
        # Initialize camera if not already running
        if not self.camera_active:
            self.camera_active = True
            self.current_model = model_type
            self.camera_task = asyncio.create_task(self.process_camera_feed())
        
        # Initialize the requested model
        await self.initialize_models(model_type)

    async def disconnect(self, websocket: WebSocket, model_type: str):
        if model_type in self.active_connections:
            self.active_connections[model_type].remove(websocket)
            if not self.active_connections[model_type]:
                del self.active_connections[model_type]
        
        # Stop camera if no more active connections
        if not self.active_connections and self.camera_active:
            self.camera_active = False
            self.camera_task.cancel()
            try:
                await self.camera_task
            except asyncio.CancelledError:
                pass

    async def process_camera_feed(self):
        cap = cv2.VideoCapture(0)
        try:
            while self.camera_active:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process for each active model type
                for model_type, connections in self.active_connections.items():
                    if not connections:
                        continue
                    
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                        
                        if model_type == 'lbph':
                            results = self._process_lbph(frame, gray, faces)
                        else:
                            results = self._process_facenet(frame, faces)
                        
                        _, buffer = cv2.imencode('.jpg', frame)
                        await asyncio.gather(*[
                            ws.send_json({
                                'frame': buffer.tobytes().hex(),
                                'results': results,
                                'model': model_type
                            }) for ws in connections
                        ])
                    except Exception as e:
                        print(f"Error processing {model_type}: {str(e)}")
                
                await asyncio.sleep(0.05)
        finally:
            cap.release()

    def _process_lbph(self, frame, gray_frame, faces):
        results = []
        for (x, y, w, h) in faces:
            try:
                face_roi = gray_frame[y:y+h, x:x+w]
                label_id, confidence = self.lbph_recognizer.recognizer.predict(face_roi)
                
                if confidence < CONFIDENCE_THRESHOLD and label_id in self.lbph_recognizer.id_to_components:
                    components = self.lbph_recognizer.id_to_components[label_id]
                    results.append({
                        'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                        'name': components['name'],
                        'emotion': components['emotion'],
                        'ethnicity': components['ethnicity'],
                        'confidence': float(confidence),
                        'recognized': True
                    })
                else:
                    results.append({
                        'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                        'confidence': float(confidence),
                        'recognized': False
                    })
            except Exception as e:
                print(f"LBPH face processing error: {str(e)}")
        return results

    def _process_facenet(self, frame, faces):
        results = []
        for (x, y, w, h) in faces:
            try:
                face_img = frame[y:y+h, x:x+w]
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_tensor = torch.FloatTensor(np.array(face_pil.resize((160, 160)))/255.*2-1)
                face_tensor = face_tensor.permute(2,0,1).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.facenet_recognizer(face_tensor).cpu().numpy().flatten()
                
                identity = self.facenet_knn.predict([embedding])[0]
                confidence = self.facenet_knn.predict_proba([embedding]).max()
                ethnicity, person, expression = identity.split('_')
                
                results.append({
                    'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                    'ethnicity': ethnicity,
                    'person': person,
                    'expression': expression,
                    'confidence': float(confidence),
                    'recognized': bool(confidence > 0.5)
                })
            except Exception as e:
                print(f"FaceNet face processing error: {str(e)}")
        return results