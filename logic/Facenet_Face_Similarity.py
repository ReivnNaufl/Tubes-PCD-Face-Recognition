from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
import numpy as np


mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_face_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        raise ValueError("Wajah tidak terdeteksi di gambar: " + img_path)
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))
    return embedding


def cosine_similarity(emb1, emb2):
    emb1_norm = emb1 / emb1.norm()
    emb2_norm = emb2 / emb2.norm()
    return torch.dot(emb1_norm.squeeze(), emb2_norm.squeeze()).item()