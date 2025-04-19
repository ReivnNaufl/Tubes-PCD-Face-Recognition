import os

PROJECT_SUB_FOLDER = "logic"
MODEL_FOLDER = "LBPH"
MODEL_FILE = os.path.join(PROJECT_SUB_FOLDER ,MODEL_FOLDER, "lbph_trained.yml")
LABELS_FILE = os.path.join(PROJECT_SUB_FOLDER, MODEL_FOLDER, "labels.json")
FACENET_MODEL_FILE = os.path.join(PROJECT_SUB_FOLDER, "Facenet", "facenet_knn.pkl")

# Camera settings
CAMERA_INDEX = 0
FRAME_RATE = 24  # Target FPS
CONFIDENCE_THRESHOLD = 85  # Lower is better for LBPH