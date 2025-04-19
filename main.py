from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from websocket import WebSocketManager
import asyncio

app = FastAPI()
ws_manager = WebSocketManager()

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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