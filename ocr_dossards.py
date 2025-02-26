import os
from PIL import Image
import pytesseract
import time

# Dossier contenant les images prétraitées
input_folder = "croppedimg"

def perform_ocr_on_images():
    processed_files = set()  # Pour suivre les fichiers déjà traités

    while True:
        # Récupérer la liste des fichiers dans le dossier
        files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

        for file in files:
            file_path = os.path.join(input_folder, file)

            if file_path not in processed_files:
                try:
                    # Charger l'image
                    img = Image.open(file_path)

                    # Effectuer l'OCR
                    text = pytesseract.image_to_string(
                        img,
                        config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
                    ).strip()

                    # Afficher le texte détecté
                    if text:
                        print(f"Fichier : {file}, Numéro détecté : {text}")

                    # Marquer le fichier comme traité
                    processed_files.add(file_path)

                except Exception as e:
                    print(f"Erreur lors du traitement du fichier {file}: {e}")

        # Attendre un moment avant de scanner à nouveau
        # time.sleep(1)

# Exécution
perform_ocr_on_images()