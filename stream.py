import os
import time
import threading
import cv2
import numpy as np
import json
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import process_images

# Charger la configuration depuis config.json
CONFIG_FILE = "config.json"

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
            print(f"Erreur lors du chargement de config.json: {e}, utilisation des valeurs par défaut")
    
    return default_config

config = load_config()

# Fichier de configuration
CONFIG_FILE = "config.json"


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
            "min_height": 600
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
                    elif isinstance(default_config[key], dict):
                        for subkey in default_config[key]:
                            if subkey not in config[key]:
                                config[key][subkey] = default_config[key][subkey]
                return config
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement de {CONFIG_FILE}: {e}")
            print(f"   Utilisation de la configuration par défaut.")
            return default_config
    else:
        print(f"⚠️  Fichier {CONFIG_FILE} introuvable. Utilisation de la configuration par défaut.")
        return default_config


# Charger la configuration
config = load_config()

# Extraire les paramètres de la configuration
RTSP_URL = config["rtsp"]["url"]
OUTPUT_FOLDER = config["folders"]["output_folder"]
CONFIDENCE_THRESHOLD = config["detection"]["confidence_threshold"]
MIN_BOX_AREA = config["detection"]["min_box_area"]
MODEL_RES = config["detection"]["model_resolution"]
MIN_HEIGHT = config["ocr"]["min_height"]

# Créer le dossier de sortie si nécessaire
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("="*60)
print("CONFIGURATION CHARGÉE")
print("="*60)
print(f"RTSP URL: {RTSP_URL}")
print(f"Dossier de sortie: {OUTPUT_FOLDER}")
print(f"Seuil de confiance: {CONFIDENCE_THRESHOLD}")
print(f"Aire minimale de box: {MIN_BOX_AREA} px")
print(f"Résolution du modèle: {MODEL_RES} px")
print(f"Hauteur minimale OCR: {MIN_HEIGHT} px")
print("="*60 + "\n")

# Charger le modèle YOLO
model = YOLO("best.pt")


class RTSPStreamReader:
    """
    Classe qui lit le flux RTSP dans un thread dédié.
    Vide le buffer système en permanence pour avoir toujours la frame la plus récente.
    """
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        """Démarre le thread de lecture."""
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def stop(self):
        """Arrête le thread de lecture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    def _update_frame(self):
        """Thread dédié : vide le buffer RTSP en continu."""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if self.cap:
                    self.cap.release()
                
                # Essayer d'abord l'URL normale
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimal
                
                # Forcer TCP via les options FFmpeg si possible
                # Le thread dédié + buffer minimal devrait suffire pour éviter le lag

                if not self.cap.isOpened():
                    time.sleep(1)
                    continue

            # Lire continuellement pour vider le buffer système
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                # Erreur de lecture, réinitialiser
                self.cap.release()
                self.cap = None
                time.sleep(0.1)

    def get_frame(self):
        """Retourne la dernière frame disponible (sans lag)."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None



def deskew_image(img):
    """
    Corrige l'inclinaison de l'image pour améliorer la reconnaissance OCR.
    """
    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return img
    
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Ne corriger que si l'angle est significatif (> 0.5 degrés)
    if abs(angle) < 0.5:
        return img
    
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), 
                             flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_for_ocr(cropped_bgr):
    """
    Prétraitement avancé optimisé pour la reconnaissance de numéros sur dossards :
    - Niveaux de gris
    - Amélioration du contraste (CLAHE)
    - Redimensionnement agressif si trop petit
    - Débruitage avancé
    - Correction d'inclinaison (deskew)
    - Binarisation Otsu (méthode la plus efficace)
    - Opérations morphologiques
    - Bordure blanche
    - Inversion si nécessaire
    
    Retourne une image binarisée prête pour l'OCR.
    """
    if len(cropped_bgr.shape) == 3:
        gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped_bgr.copy()

    # Redimensionnement si trop petit (upscale pour meilleure qualité)
    h, w = gray.shape
    if h < MIN_HEIGHT:
        scale = MIN_HEIGHT / h
        gray = cv2.resize(
            gray, (int(w * scale), MIN_HEIGHT),
            interpolation=cv2.INTER_CUBIC
        )

    # Amélioration du contraste local avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Débruitage avancé
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Correction d'inclinaison
    gray = deskew_image(gray)

    # Binarisation Otsu avec blur
    gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        gray_blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Opérations morphologiques pour nettoyer le bruit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Tesseract attend du texte noir sur fond blanc
    if np.mean(binary) < 127:
        binary = 255 - binary
    
    # Ajouter une bordure blanche (Tesseract marche mieux)
    binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, 
                                cv2.BORDER_CONSTANT, value=255)

    return binary


def process_frame(frame):
    """
    Détecte les dossards dans la frame avec YOLO en haute résolution.
    Retourne la frame avec les bounding boxes dessinées.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # YOLO en 1280px pour meilleure précision
    results = model(img_rgb, verbose=False, imgsz=MODEL_RES)

    frame_with_boxes = frame.copy()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, confidence in zip(boxes, confidences):
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            box_area = (x2 - x1) * (y2 - y1)

            # Couleur de la bounding box : vert si valide, orange si trop petite
            is_valid = box_area >= MIN_BOX_AREA
            if box_area < MIN_BOX_AREA:
                color = (0, 165, 255)  # Orange : trop petite
            else:
                color = (0, 255, 0)  # Vert : valide
            thickness = 2

            # Dessiner la bounding box
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, thickness)

            # Label avec confiance et aire
            status = "OK" if is_valid else "Trop petit"
            label = f"Conf: {confidence:.2f} | Aire: {box_area} | {status}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame_with_boxes,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                frame_with_boxes,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # Enregistrer seulement les détections valides
            if not is_valid:
                continue

            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            # Préprocesser avec le pipeline avancé (prêt pour OCR)
            preprocessed = preprocess_for_ocr(cropped)

            # Enregistrer en PNG sans perte (important pour OCR)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            microseconds = int(time.time() * 1000000) % 1000000
            filename = os.path.join(OUTPUT_FOLDER, f"dossard_{timestamp}_{microseconds}.png")
            cv2.imwrite(filename, preprocessed)
            print(f"Dossard préprocessé enregistré: {filename} (aire: {box_area}, confiance: {confidence:.2f})")

    return frame_with_boxes


def run_rtsp_detection():
    """
    Lance la détection avec thread dédié pour la lecture RTSP.
    """
    print("Connexion au flux RTSP...")
    print("Détection des dossards en cours (Ctrl+C pour arrêter)...\n")

    stream_reader = RTSPStreamReader(RTSP_URL)
    stream_reader.start()

    # Attendre que la première frame soit disponible
    time.sleep(1)

    try:
        while True:
            frame = stream_reader.get_frame()

            if frame is None:
                time.sleep(0.1)
                continue

            # Traiter la frame avec YOLO
            frame_with_boxes = process_frame(frame)

            # Afficher la frame avec les bounding boxes
            cv2.imshow("Détection de dossards - RTSP", frame_with_boxes)

            # Quitter avec 'q' ou 'ESC'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\nArrêt du script.")
    finally:
        stream_reader.stop()
        cv2.destroyAllWindows()
        print("Connexion fermée.")


if __name__ == "__main__":
    run_rtsp_detection()
