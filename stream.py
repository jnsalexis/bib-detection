import os
import time
import threading
import cv2
from datetime import datetime
from ultralytics import YOLO

# URL du flux RTSP
RTSP_URL = "rtsp://admin:teamprod123@192.168.70.101:554/h264Preview_01_main"

# Dossier de sortie pour les images de dossards détectés
OUTPUT_FOLDER = "img"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Seuil de confiance YOLO
CONFIDENCE_THRESHOLD = 0.3

# Aire minimale de la bounding box (en pixels) pour garantir une image lisible
MIN_BOX_AREA = 1000

# Résolution YOLO : 1280px pour meilleure précision sur petits dossards
MODEL_RES = 1280

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



def preprocess_to_grayscale(cropped_bgr):
    """Préprocesse le crop en noir et blanc (niveaux de gris)."""
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    return gray


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

            # Préprocesser en noir et blanc
            preprocessed = preprocess_to_grayscale(cropped)

            # Enregistrer avec compression JPEG qualité 100 (zéro perte)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUTPUT_FOLDER, f"dossard_{timestamp}_{int(box_area)}.jpg")
            cv2.imwrite(filename, preprocessed, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"Dossard enregistré: {filename} (aire: {box_area}, confiance: {confidence:.2f})")

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
