import os
import shutil
import uuid
from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torch import cosine_similarity
from logic.Facenet_Face_Similarity import get_face_embedding
from websocket import WebSocketManager
import asyncio

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = FastAPI()
ws_manager = WebSocketManager()

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/lbph")
async def index(request: Request):
    return templates.TemplateResponse("lbph.html", {"request": request})

@app.get("/facenet")
async def index(request: Request):
    return templates.TemplateResponse("facenet.html", {"request": request})

@app.websocket("/lbph/ws")
async def lbph_websocket(websocket: WebSocket):
    await ws_manager.connect(websocket, 'lbph')
    try:
        while True:
            # Just keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, 'lbph')

@app.websocket("/facenet/ws")
async def facenet_websocket(websocket: WebSocket):
    await ws_manager.connect(websocket, 'facenet')
    try:
        while True:
            # Just keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, 'facenet')

@app.get("/similarity/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

def delete_files(path1, path2):
    import time
    time.sleep(5)  # beri waktu 5 detik agar browser bisa ambil gambarnya
    os.remove(path1)
    os.remove(path2)

@app.post("/similarity/", response_class=HTMLResponse)
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    filename1 = f"{uuid.uuid4().hex}_{file1.filename}"
    filename2 = f"{uuid.uuid4().hex}_{file2.filename}"

    path1 = os.path.join("uploads", filename1)
    path2 = os.path.join("uploads", filename2)

    with open(path1, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    with open(path2, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    try:
        emb1 = get_face_embedding(path1)
        emb2 = get_face_embedding(path2)
        score = cosine_similarity(emb1, emb2).item()

        response = templates.TemplateResponse("upload.html", {
            "request": request,
            "similarity_score": score,
            "image1_url": f"/uploads/{filename1}",
            "image2_url": f"/uploads/{filename2}"
        })

        # Tambahkan ke background task
        background_tasks.add_task(delete_files, path1, path2)
        return response

    except Exception as e:
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "error": str(e)
        })