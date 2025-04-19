import os

PROJECT_SUB_FOLDER = "logic"
MODEL_FOLDER = "LBPH"
MODEL_FILE = os.path.join(PROJECT_SUB_FOLDER ,MODEL_FOLDER, "lbph_trained.yml")
LABELS_FILE = os.path.join(PROJECT_SUB_FOLDER, MODEL_FOLDER, "labels.json")

# Camera settings
CAMERA_INDEX = 0
FRAME_RATE = 24  # Target FPS
CONFIDENCE_THRESHOLD = 85  # Lower is better for LBPH