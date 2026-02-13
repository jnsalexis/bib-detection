from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import os
import json
import subprocess
import signal
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Optional as Opt

# Charger les variables d'environnement
load_dotenv()

app = FastAPI(title="Bib Detection - Admin Panel")
templates = Jinja2Templates(directory="templates")

# Servir les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

CONFIG_FILE = "config.json"

# Stockage des processus en cours
running_processes = {
    "stream": None,
    "process_images": None
}

# Configuration Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "dossards")

# Client Supabase
_supabase_client: Opt[Client] = None


def load_config():
    """Charge la configuration depuis config.json."""
    default_config = {
        "detection": {
            "confidence_threshold": 0.3,
            "min_box_area": 1000,
            "model_resolution": 1280,
            "required_detections": 3
        },
        "ocr": {
            "min_height": 400
        },
        "rtsp": {
            "url": "rtsp://admin:teamprod123@192.168.70.101:554/h264Preview_01_main"
        },
        "folders": {
            "output_folder": "img",
            "processed_folder": "img_processed"
        }
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                # Fusionner avec les valeurs par défaut pour les nouvelles clés
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        except Exception as e:
            print(f"Erreur lors du chargement de config.json: {e}")
    
    return default_config


def save_config(config):
    """Sauvegarde la configuration dans config.json."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de config.json: {e}")
        return False


def get_supabase_client() -> Opt[Client]:
    """Retourne le client Supabase."""
    global _supabase_client
    
    api_key = SUPABASE_SERVICE_ROLE_KEY if SUPABASE_SERVICE_ROLE_KEY else SUPABASE_KEY
    
    if not SUPABASE_URL or not api_key:
        return None
    
    if _supabase_client is None:
        try:
            _supabase_client = create_client(SUPABASE_URL, api_key)
        except Exception as e:
            print(f"Erreur lors de la création du client Supabase: {e}")
            return None
    
    return _supabase_client


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Page d'accueil avec le tableau des données."""
    client = get_supabase_client()
    dossards = []
    error = None
    
    if client:
        try:
            response = client.table(SUPABASE_TABLE).select("*").order("created_at", desc=True).limit(1000).execute()
            dossards = response.data if response.data else []
        except Exception as e:
            error = f"Erreur lors de la récupération des données: {str(e)}"
    else:
        error = "Supabase non configuré. Vérifiez vos variables d'environnement."
    
    config = load_config()
    
    # Vérifier l'état des processus
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
    """Page de configuration des paramètres de détection."""
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
    """Met à jour le fichier config.json avec les nouvelles valeurs."""
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


@app.get("/api/dossards")
async def get_dossards():
    """API endpoint pour récupérer les dossards."""
    client = get_supabase_client()
    
    if not client:
        return {"error": "Supabase non configuré", "data": []}
    
    try:
        response = client.table(SUPABASE_TABLE).select("*").order("created_at", desc=True).limit(1000).execute()
        return {"error": None, "data": response.data if response.data else []}
    except Exception as e:
        return {"error": str(e), "data": []}


@app.delete("/api/dossards/{dossard_id}")
async def delete_dossard(dossard_id: int):
    """Supprime un dossard de Supabase."""
    client = get_supabase_client()
    
    if not client:
        return {"error": "Supabase non configuré", "success": False}
    
    try:
        client.table(SUPABASE_TABLE).delete().eq("id", dossard_id).execute()
        return {"error": None, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


@app.post("/api/scripts/start/{script_name}")
async def start_script(script_name: str):
    """Démarre un script Python."""
    if script_name not in ["stream", "process_images"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Script invalide", "success": False}
        )
    
    # Vérifier si le script est déjà en cours d'exécution
    if running_processes[script_name] is not None:
        if running_processes[script_name].poll() is None:
            return JSONResponse(
                status_code=400,
                content={"error": f"{script_name}.py est déjà en cours d'exécution", "success": False}
            )
    
    try:
        script_file = f"{script_name}.py"
        if not os.path.exists(script_file):
            return JSONResponse(
                status_code=404,
                content={"error": f"Fichier {script_file} introuvable", "success": False}
            )
        
        # Lancer le script en arrière-plan
        process = subprocess.Popen(
            ["python", script_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        
        running_processes[script_name] = process
        
        return {"error": None, "success": True, "message": f"{script_name}.py démarré avec succès"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "success": False}
        )


@app.post("/api/scripts/stop/{script_name}")
async def stop_script(script_name: str):
    """Arrête un script Python."""
    if script_name not in ["stream", "process_images"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Script invalide", "success": False}
        )
    
    if running_processes[script_name] is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"{script_name}.py n'est pas en cours d'exécution", "success": False}
        )
    
    try:
        process = running_processes[script_name]
        
        # Vérifier si le processus est toujours actif
        if process.poll() is None:
            # Arrêter le processus
            if os.name == 'nt':  # Windows
                process.terminate()
            else:  # Unix/Linux/Mac
                process.send_signal(signal.SIGTERM)
            
            # Attendre un peu puis forcer l'arrêt si nécessaire
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            running_processes[script_name] = None
            return {"error": None, "success": True, "message": f"{script_name}.py arrêté avec succès"}
        else:
            running_processes[script_name] = None
            return {"error": None, "success": True, "message": f"{script_name}.py était déjà arrêté"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "success": False}
        )


@app.get("/api/scripts/status")
async def get_scripts_status():
    """Retourne l'état des scripts."""
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


if __name__ == "__main__":
    import uvicorn
    print("🚀 Démarrage de l'interface web...")
    print("🌐 Interface disponible sur http://localhost:8001")
    print("📊 Tableau des données: http://localhost:8001")
    print("⚙️  Configuration: http://localhost:8001/config")
    uvicorn.run(app, host="0.0.0.0", port=8001)
