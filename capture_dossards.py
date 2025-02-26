import cv2
from datetime import datetime
import os
from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO("best.pt")

# Créez un dossier pour enregistrer les images si ce n'est pas déjà fait
output_folder = "croppedimg"
os.makedirs(output_folder, exist_ok=True)

# Définir les seuils de taille des boîtes pour filtrer les images trop petites
MIN_BOX_AREA = 10000

def process_frame(frame):
    # Convertir la frame en RGB pour YOLOv8
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Effectuer la détection du dossard
    results = model(img_rgb, verbose=False)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()  # Confiance

        for box, confidence in zip(boxes, confidences):
            if confidence > 0.7:  # Seuil de confiance YOLO
                x1, y1, x2, y2 = map(int, box)

                # Calculer l'aire de la boîte englobante
                box_area = (x2 - x1) * (y2 - y1)

                # Filtrer les boîtes trop petites
                if box_area > MIN_BOX_AREA:
                    # Recadrer la zone détectée
                    cropped = frame[y1:y2, x1:x2]

                    # Convertir en niveaux de gris
                    binary = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    # Appliquer le seuillage adaptatif
                    binary = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 19, 6)

                    # Enregistrer l'image prétraitée dans le dossier

                    filename = f"{output_folder}/dossard_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
                    cv2.imwrite(filename, binary)
                    print(f"Image enregistrée : {filename}")

def process_webcam():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Traiter la frame et enregistrer les images détectées
        process_frame(frame)

        # Afficher la frame en direct
        cv2.imshow('Dossard Detection - YOLOv8', frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Exécution
process_webcam()