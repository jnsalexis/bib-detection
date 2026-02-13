#=================================================================================
# IMPORTS
#=================================================================================
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from typing import Optional
import os
import json
import subprocess
import signal
import cv2
import numpy as np
import threading
import time
from dotenv import load_dotenv
from supabase import create_client, Client


#=================================================================================
# CONFIGURATION
#=================================================================================
load_dotenv()

app = FastAPI(title="Bib Detection - Admin Panel")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

CONFIG_FILE = "config.json"

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "dossards")


#=================================================================================
# VARIABLES GLOBALES
#=================================================================================
running_processes = {
    "stream": None,
    "process_images": None
}

_supabase_client: Optional[Client] = None
_video_stream = None
_stream_lock = threading.Lock()


#=================================================================================
# CLASSES
#=================================================================================
class RTSPVideoStream:
    """Gestion du flux RTSP pour le streaming web."""
    
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.last_access = time.time()
    
    def start(self):
        """Demarre le thread de lecture du flux."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Arrete le thread de lecture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _update_frame(self):
        """Boucle de lecture des frames."""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if self.cap:
                    self.cap.release()
                
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not self.cap.isOpened():
                    time.sleep(1)
                    continue
            
            ret, frame = self.cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                if height > 720:
                    scale = 720 / height
                    new_width = int(width * scale)
                    frame = cv2.resize(frame, (new_width, 720))
                
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.last_access = time.time()
            else:
                self.cap.release()
                self.cap = None
                time.sleep(0.1)
    
    def get_frame(self):
        """Retourne la derniere frame disponible."""
        with self.lock:
            self.last_access = time.time()
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def is_active(self):
        """Verifie si le stream a ete accede recemment."""
        return (time.time() - self.last_access) < 60


#=================================================================================
# FONCTIONS - CONFIGURATION
#=================================================================================
def load_config():
    """Charge la configuration depuis config.json."""
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Fichier {CONFIG_FILE} introuvable.")
    
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(config):
    """Sauvegarde la configuration dans config.json."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        return False


#=================================================================================
# FONCTIONS - SUPABASE
#=================================================================================
def get_supabase_client() -> Optional[Client]:
    """Retourne le client Supabase."""
    global _supabase_client
    
    api_key = SUPABASE_SERVICE_ROLE_KEY if SUPABASE_SERVICE_ROLE_KEY else SUPABASE_KEY
    
    if not SUPABASE_URL or not api_key:
        return None
    
    if _supabase_client is None:
        try:
            _supabase_client = create_client(SUPABASE_URL, api_key)
        except Exception as e:
            print(f"Erreur creation client Supabase: {e}")
            return None
    
    return _supabase_client


#=================================================================================
# ROUTES - PAGES
#=================================================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Page d'accueil avec le tableau des donnees."""
    client = get_supabase_client()
    dossards = []
    error = None
    
    if client:
        try:
            response = client.table(SUPABASE_TABLE).select("*").order("created_at", desc=True).limit(1000).execute()
            dossards = response.data if response.data else []
        except Exception as e:
            error = f"Erreur lors de la recuperation des donnees: {str(e)}"
    else:
        error = "Supabase non configure."
    
    config = load_config()
    
    processes_status = {
        "stream": running_processes["stream"] is not None and running_processes["stream"].poll() is None,
        "process_images": running_processes["process_images"] is not None and running_processes["process_images"].poll() is None
    }
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "dossards": dossards,
        "error": error,
        "detection_config": config,
        "processes_status": processes_status
    })


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    """Page de configuration."""
    config = load_config()
    return templates.TemplateResponse("config.html", {
        "request": request,
        "config": config
    })


@app.post("/config/update")
async def update_config(
    confidence_threshold: float = Form(...),
    min_box_area: int = Form(...),
    model_resolution: int = Form(...),
    required_detections: int = Form(...),
    min_height: int = Form(...),
    rtsp_url: str = Form(...),
    output_folder: str = Form(...),
    processed_folder: str = Form(...)
):
    """Met a jour config.json."""
    config = {
        "detection": {
            "confidence_threshold": confidence_threshold,
            "min_box_area": min_box_area,
            "model_resolution": model_resolution,
            "required_detections": required_detections
        },
        "ocr": {
            "min_height": min_height
        },
        "rtsp": {
            "url": rtsp_url
        },
        "folders": {
            "output_folder": output_folder,
            "processed_folder": processed_folder
        }
    }
    
    if save_config(config):
        return RedirectResponse(url="/config?success=1", status_code=303)
    else:
        return RedirectResponse(url="/config?error=1", status_code=303)


#=================================================================================
# ROUTES - API DOSSARDS
#=================================================================================
@app.get("/api/dossards")
async def get_dossards():
    """Recupere les dossards."""
    client = get_supabase_client()
    
    if not client:
        return {"error": "Supabase non configure", "data": []}
    
    try:
        response = client.table(SUPABASE_TABLE).select("*").order("created_at", desc=True).limit(1000).execute()
        return {"error": None, "data": response.data if response.data else []}
    except Exception as e:
        return {"error": str(e), "data": []}


@app.delete("/api/dossards/{dossard_id}")
async def delete_dossard(dossard_id: int):
    """Supprime un dossard."""
    client = get_supabase_client()
    
    if not client:
        return {"error": "Supabase non configure", "success": False}
    
    try:
        client.table(SUPABASE_TABLE).delete().eq("id", dossard_id).execute()
        return {"error": None, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


#=================================================================================
# ROUTES - API SCRIPTS
#=================================================================================
@app.post("/api/scripts/start/{script_name}")
async def start_script(script_name: str):
    """Demarre un script Python."""
    if script_name not in ["stream", "process_images"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Script invalide", "success": False}
        )
    
    if running_processes[script_name] is not None:
        if running_processes[script_name].poll() is None:
            return JSONResponse(
                status_code=400,
                content={"error": f"{script_name}.py deja en cours", "success": False}
            )
    
    try:
        script_file = f"{script_name}.py"
        if not os.path.exists(script_file):
            return JSONResponse(
                status_code=404,
                content={"error": f"Fichier {script_file} introuvable", "success": False}
            )
        
        process = subprocess.Popen(
            ["python", script_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        
        running_processes[script_name] = process
        
        return {"error": None, "success": True, "message": f"{script_name}.py demarre"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "success": False}
        )


@app.post("/api/scripts/stop/{script_name}")
async def stop_script(script_name: str):
    """Arrete un script Python."""
    if script_name not in ["stream", "process_images"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Script invalide", "success": False}
        )
    
    if running_processes[script_name] is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"{script_name}.py non demarre", "success": False}
        )
    
    try:
        process = running_processes[script_name]
        
        if process.poll() is None:
            if os.name == 'nt':
                process.terminate()
            else:
                process.send_signal(signal.SIGTERM)
            
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            running_processes[script_name] = None
            return {"error": None, "success": True, "message": f"{script_name}.py arrete"}
        else:
            running_processes[script_name] = None
            return {"error": None, "success": True, "message": f"{script_name}.py deja arrete"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "success": False}
        )


@app.get("/api/scripts/status")
async def get_scripts_status():
    """Retourne l'etat des scripts."""
    status = {}
    for script_name in ["stream", "process_images"]:
        process = running_processes[script_name]
        if process is not None:
            status[script_name] = {
                "running": process.poll() is None,
                "returncode": process.returncode if process.poll() is not None else None
            }
        else:
            status[script_name] = {"running": False, "returncode": None}
    
    return {"status": status}

#=================================================================================
# MAIN
#=================================================================================
if __name__ == "__main__":
    import uvicorn
    print("Demarrage de l'interface web...")
    print("Interface disponible sur http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
