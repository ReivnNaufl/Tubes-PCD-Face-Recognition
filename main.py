from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from websocket import WebSocketManager

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

@app.websocket("/lbph/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        await ws_manager.broadcast_frame(websocket)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)