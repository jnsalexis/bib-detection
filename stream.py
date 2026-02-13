#=================================================================================
# IMPORTS
#=================================================================================
import os
import time
import threading
import cv2
import numpy as np
import json
from datetime import datetime
from ultralytics import YOLO


#=================================================================================
# VARIABLES - CONFIGURATION
#=================================================================================
def load_config():
    """Charge la configuration depuis config.json."""
    if not os.path.exists("config.json"):
        raise FileNotFoundError("Fichier config.json introuvable.")
    
    with open("config.json", "r") as f:
        return json.load(f)


config = load_config()

RTSP_URL = config["rtsp"]["url"]
OUTPUT_FOLDER = config["folders"]["output_folder"]
CONFIDENCE_THRESHOLD = config["detection"]["confidence_threshold"]
MIN_BOX_AREA = config["detection"]["min_box_area"]
MODEL_RES = config["detection"]["model_resolution"]
MIN_HEIGHT = config["ocr"]["min_height"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("="*60)
print("CONFIGURATION")
print("="*60)
print(f"RTSP URL: {RTSP_URL}")
print(f"Dossier de sortie: {OUTPUT_FOLDER}")
print(f"Seuil de confiance: {CONFIDENCE_THRESHOLD}")
print(f"Aire minimale: {MIN_BOX_AREA} px")
print(f"Resolution modele: {MODEL_RES} px")
print(f"Hauteur minimale OCR: {MIN_HEIGHT} px")
print("="*60 + "\n")

model = YOLO("best.pt")


#=================================================================================
# CLASSES
#=================================================================================
class RTSPStreamReader:
    """Lecture du flux RTSP dans un thread dedie."""
    
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        """Demarre le thread de lecture."""
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
                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                self.cap.release()
                self.cap = None
                time.sleep(0.1)

    def get_frame(self):
        """Retourne la derniere frame disponible."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None


#=================================================================================
# FONCTIONS - PREPROCESSING
#=================================================================================
def deskew_image(img):
    """Corrige l'inclinaison de l'image."""
    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return img
    
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
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
    """Pipeline de preprocessing pour OCR."""
    if len(cropped_bgr.shape) == 3:
        gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped_bgr.copy()

    h, w = gray.shape
    if h < MIN_HEIGHT:
        scale = MIN_HEIGHT / h
        gray = cv2.resize(gray, (int(w * scale), MIN_HEIGHT), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray = deskew_image(gray)

    gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    if np.mean(binary) < 127:
        binary = 255 - binary
    
    binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    return binary


#=================================================================================
# FONCTIONS - DETECTION
#=================================================================================
def process_frame(frame):
    """Detecte les dossards avec YOLO et les sauvegarde."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

            is_valid = box_area >= MIN_BOX_AREA
            color = (0, 255, 0) if is_valid else (0, 165, 255)

            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)

            status = "OK" if is_valid else "Trop petit"
            label = f"Conf: {confidence:.2f} | Aire: {box_area} | {status}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_with_boxes, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if not is_valid:
                continue

            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            preprocessed = preprocess_for_ocr(cropped)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            microseconds = int(time.time() * 1000000) % 1000000
            filename = os.path.join(OUTPUT_FOLDER, f"dossard_{timestamp}_{microseconds}.png")
            cv2.imwrite(filename, preprocessed)
            print(f"Dossard enregistre: {filename} (aire: {box_area}, conf: {confidence:.2f})")

    return frame_with_boxes


def run_rtsp_detection():
    """Lance la detection en temps reel."""
    print("Connexion au flux RTSP...")
    print("Detection en cours (Ctrl+C pour arreter)\n")

    stream_reader = RTSPStreamReader(RTSP_URL)
    stream_reader.start()
    time.sleep(1)

    try:
        while True:
            frame = stream_reader.get_frame()

            if frame is None:
                time.sleep(0.1)
                continue

            frame_with_boxes = process_frame(frame)
            cv2.imshow("Detection de dossards - RTSP", frame_with_boxes)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\nArret du script.")
    finally:
        stream_reader.stop()
        cv2.destroyAllWindows()
        print("Connexion fermee.")


#=================================================================================
# MAIN
#=================================================================================
if __name__ == "__main__":
    run_rtsp_detection()
