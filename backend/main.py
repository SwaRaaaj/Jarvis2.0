import asyncio
import json
import os
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from memory_vault import MemoryVault
from pc_telemetry import PCTelemetry
from screen_vision import ScreenVision
from os_automation import OSAutomation
from voice_engine import VoiceEngine
from ollama_engine import OllamaEngine

app = FastAPI(title="JARVIS AI Brain Core", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = MemoryVault()
vision = ScreenVision()
voice = VoiceEngine()
ollama = OllamaEngine(memory_vault=memory, screen_vision=vision)

class CommandRequest(BaseModel):
    command: str
    model: str = "llama-3.3-70b-versatile"
    speak_response: bool = True

class MemoryUpdateRequest(BaseModel):
    key: str
    value: str

class SpeakRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "status": "online",
        "system": "JARVIS AI Brain Core Server",
        "ollama_models": ollama.list_local_models(),
        "user_name": memory.get_user_info("user_name") or "Boss"
    }

@app.get("/api/telemetry/specs")
def get_specs():
    return PCTelemetry.get_system_specs()

@app.get("/api/telemetry/live")
def get_live_metrics():
    return PCTelemetry.get_live_metrics()

@app.get("/api/models")
def get_models():
    return {"models": ollama.list_local_models()}

@app.get("/api/agents")
def get_agent_stats():
    """Per-agent instrumentation: model calls, cache hit rates, grounding methods, learned rules."""
    return {"engine": "cortex" if ollama.cortex is not None else "legacy", "agents": ollama.agent_stats()}

@app.get("/api/memory")
def get_memory():
    return {
        "profile": memory.get_all_user_info(),
        "recent_logs": memory.get_recent_logs(15)
    }

@app.post("/api/memory")
def update_memory(req: MemoryUpdateRequest):
    memory.set_user_info(req.key, req.value)
    return {"status": "updated", "key": req.key, "value": req.value}

@app.post("/api/tts")
def trigger_tts(req: SpeakRequest):
    voice.speak(req.text)
    return {"status": "speaking", "text": req.text}

@app.post("/api/cancel")
def cancel_task():
    ollama.cancel()
    voice.stop()
    return {"status": "cancelled"}

@app.post("/api/execute")
def execute_command(req: CommandRequest):
    responses = []
    for item in ollama.process_command(req.command, model_override=req.model):
        responses.append(item)
        if item.get("type") == "response" and req.speak_response:
            voice.speak(item.get("text"))
    return {"events": responses}

# WebSocket Manager for Real-time Dashboard Sync
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Background Telemetry & Screen Perception Broadcaster Loop
async def telemetry_broadcaster():
    while True:
        if manager.active_connections:
            try:
                live_metrics = PCTelemetry.get_live_metrics()
                screen_frame = vision.capture_base64_jpeg(quality=40, scale=0.35)
                await manager.broadcast({
                    "type": "telemetry_tick",
                    "telemetry": live_metrics,
                    "screen_frame": screen_frame,
                    "is_speaking": voice.is_speaking
                })
            except Exception as e:
                pass
        await asyncio.sleep(0.5)  # 2 Hz update rate

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(telemetry_broadcaster())
    voice.speak("JARVIS AI Brain online and operational, Boss.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "init",
            "specs": PCTelemetry.get_system_specs(),
            "memory": memory.get_all_user_info(),
            "models": ollama.list_local_models()
        })
    except Exception:
        manager.disconnect(websocket)
        return
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                action_type = payload.get("type")

                if action_type == "voice_command" or action_type == "text_command":
                    user_cmd = payload.get("command", "")
                    model_choice = payload.get("model", "llama-3.3-70b-versatile")
                    speak_out = payload.get("speak", True)

                    await websocket.send_json({"type": "status", "message": f"Processing: '{user_cmd}'"})

                    loop = asyncio.get_event_loop()
                    def run_ollama():
                        return list(ollama.process_command(user_cmd, model_override=model_choice))

                    events = await loop.run_in_executor(None, run_ollama)

                    for event in events:
                        await websocket.send_json(event)
                        if event.get("type") == "response" and speak_out:
                            voice.speak(event.get("text"))

                elif action_type == "quick_action":
                    action_name = payload.get("action")
                    if action_name == "stop_speech":
                        voice.stop()
                        await websocket.send_json({"type": "status", "message": "Speech halted."})
                    elif action_name == "screenshot_grid":
                        grid_b64, grid_meta = vision.capture_grid_overlay_base64()
                        await websocket.send_json({"type": "grid_overlay", "image": grid_b64, "grid": grid_meta})
                    elif action_name == "launch_app":
                        res = OSAutomation.launch_app(payload.get("app_name", "notepad"))
                        await websocket.send_json({"type": "tool_exec", "output": res})
                    elif action_name == "cancel_task":
                        ollama.cancel()
                        await websocket.send_json({"type": "status", "message": "Task cancellation requested."})

            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON format."})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
