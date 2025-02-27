import os
from PIL import Image
import pytesseract

# Dossier contenant les images prétraitées
input_folder = "croppedimg"

def perform_ocr_on_images():
    # Analyse toutes les images du dossier une seule fois.
    files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

    count = 0

    for file in files:
        file_path = os.path.join(input_folder, file)

        if file_path:
            try:
                if not os.path.isfile(file_path):
                    print(f"Fichier introuvable : {file}")
                    continue

                # Charger l'image
                img = Image.open(file_path)

                # Effectuer l'OCR
                text = pytesseract.image_to_string(
                    img,
                    config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
                ).strip()

                # Afficher le texte détecté
                if text:
                    print(f"{file} → Numéro détecté : {text}")

                count += 1

            except Exception as e:
                print(f"Erreur lors du traitement de {file}: {e}")

perform_ocr_on_images()