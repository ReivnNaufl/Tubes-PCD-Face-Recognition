import cv2
import asyncio
from typing import List
from fastapi import WebSocket
from logic.LBPH_Face_Recognition import FaceRecognizer
from config import CAMERA_INDEX, FRAME_RATE

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.recognizer = FaceRecognizer()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast_frame(self, websocket: WebSocket):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        frame_delay = 1 / FRAME_RATE
        
        try:
            while True:
                start_time = asyncio.get_event_loop().time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                recognition_results = self.recognizer.recognize_faces(frame)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                await websocket.send_json({
                    "frame": frame_bytes.hex(),
                    "results": recognition_results
                })
                
                # Maintain target frame rate
                processing_time = asyncio.get_event_loop().time() - start_time
                await asyncio.sleep(max(0, frame_delay - processing_time))
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()
            self.disconnect(websocket)