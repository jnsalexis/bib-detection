import os
import re
import cv2
import numpy as np
from PIL import Image
import pytesseract
from collections import defaultdict
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Dossier contenant les images à analyser
IMG_FOLDER = "img"
# Dossier pour enregistrer les images préprocessées (ce que l'OCR analyse)
IMG_PROCESSED_FOLDER = "img_processed"
os.makedirs(IMG_PROCESSED_FOLDER, exist_ok=True)

# Taille minimale (hauteur) pour que Tesseract lise bien les chiffres
MIN_HEIGHT = 400

# Configurations Tesseract pour les dossards (chiffres uniquement)
CONFIG_PSM8 = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
CONFIG_PSM7 = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
CONFIG_PSM6 = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"

# Configuration Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# Utiliser SUPABASE_SERVICE_ROLE_KEY si disponible (contourne RLS), sinon SUPABASE_KEY
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "dossards")

# Nombre de détections requises pour envoyer à Supabase
REQUIRED_DETECTIONS = 3

# Client Supabase (initialisé à None, sera créé à la première utilisation)
_supabase_client: Optional[Client] = None

# Compteur des détections par numéro
detection_counts = defaultdict(int)
# Numéros déjà envoyés à Supabase (pour éviter les doublons)
sent_numbers = set()
# Numéros pour lesquels l'envoi a échoué (pour éviter de réessayer indéfiniment)
failed_numbers = set()


def preprocess_for_ocr(img):
    """
    Prétraitement optimisé pour la reconnaissance de numéros sur dossards :
    - Niveaux de gris
    - Redimensionnement si trop petit
    - Binarisation Otsu (chiffres noirs sur fond blanc)
    - Inversion si nécessaire
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape
    if h < MIN_HEIGHT:
        scale = MIN_HEIGHT / h
        gray = cv2.resize(
            gray, (int(w * scale), MIN_HEIGHT),
            interpolation=cv2.INTER_CUBIC
        )

    # Léger flou pour réduire le bruit sans perdre les bords des chiffres
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarisation Otsu : séparation nette chiffres / fond
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Tesseract attend du texte noir sur fond blanc
    if np.mean(binary) < 127:
        binary = 255 - binary

    return binary


def get_supabase_client() -> Optional[Client]:
    """
    Retourne le client Supabase, ou None si les credentials ne sont pas configurés.
    Utilise SUPABASE_SERVICE_ROLE_KEY si disponible (contourne RLS), sinon SUPABASE_KEY.
    """
    global _supabase_client
    
    # Préférer la service_role key qui contourne RLS
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


def save_number_to_supabase(number: str) -> bool:
    """
    Enregistre un numéro validé dans Supabase (table dossards).
    La table a les colonnes: id (auto), created_at (auto), number (int8)
    
    Args:
        number: Le numéro de dossard détecté (sera converti en int)
    
    Returns:
        True si l'enregistrement a réussi, False sinon
    """
    client = get_supabase_client()
    
    if client is None:
        print("Supabase non configuré (SUPABASE_URL et SUPABASE_KEY requis)")
        return False
    
    try:
        # Convertir le numéro en entier pour correspondre au type int8 de la table
        number_int = int(number)
        
        # Vérifier si le numéro existe déjà
        existing = client.table(SUPABASE_TABLE).select("id").eq("number", number_int).execute()
        
        if existing.data:
            # Le numéro existe déjà, on ne fait rien (ou on peut logger)
            print(f"Numéro {number_int} existe déjà dans Supabase")
            return True
        else:
            # Créer un nouvel enregistrement
            # created_at sera automatiquement rempli par Supabase
            result = client.table(SUPABASE_TABLE).insert({
                "number": number_int
            }).execute()
            
            print(f"✅ Numéro {number_int} enregistré dans Supabase (table: {SUPABASE_TABLE})")
            return True
        
    except ValueError:
        print(f"❌ Erreur: '{number}' n'est pas un numéro valide")
        return False
    except Exception as e:
        error_msg = str(e)
        if "row-level security policy" in error_msg.lower() or "42501" in error_msg:
            print(f"❌ Erreur RLS (Row Level Security): L'insertion est bloquée par les politiques de sécurité.")
            print(f"   Solutions possibles:")
            print(f"   1. Utiliser SUPABASE_SERVICE_ROLE_KEY dans votre .env (recommandé pour les scripts backend)")
            print(f"   2. Créer une politique RLS dans Supabase qui permet l'INSERT sur la table 'dossards'")
            print(f"   3. Désactiver temporairement RLS sur la table (non recommandé en production)")
        else:
            print(f"❌ Erreur lors de l'enregistrement dans Supabase: {e}")
        return False


def ocr_image(img_binary):
    """
    Lance l'OCR sur une image binarisée avec plusieurs PSM et retourne
    le meilleur résultat (texte brut et numéro extrait).
    """
    pil = Image.fromarray(img_binary)
    results = []

    for config in (CONFIG_PSM8, CONFIG_PSM7, CONFIG_PSM6):
        try:
            raw = pytesseract.image_to_string(pil, config=config).strip()
            digits = "".join(c for c in raw if c.isdigit())
            if digits and not (len(digits) == 4 and digits.startswith("20")):
                results.append((raw, digits))
        except Exception:
            continue

    if not results:
        return ("", "")

    # Prendre le résultat avec le plus long numéro (souvent le bon dossard)
    best = max(results, key=lambda x: len(x[1]))
    return best


def process_image_file(image_path):
    """
    Charge une image, la préprocesse et effectue l'OCR.
    Enregistre l'image préprocessée dans img_processed pour avoir un retour visuel.
    
    Returns:
        (raw_text, numbers_str)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ("", "")
        binary = preprocess_for_ocr(img)

        # Enregistrer l'image préprocessée dans img_processed (ce que l'OCR analyse)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        processed_filename = os.path.join(IMG_PROCESSED_FOLDER, f"{base_name}_preprocessed.png")
        cv2.imwrite(processed_filename, binary)

        raw, digits = ocr_image(binary)
        return (raw, digits)
    except Exception:
        return ("", "")


def run_ocr_on_img_folder():
    """
    Effectue l'OCR sur toutes les images du répertoire img.
    Compte les occurrences de chaque numéro et envoie à Supabase quand un numéro est détecté 3 fois.
    """
    global detection_counts, sent_numbers, failed_numbers
    
    # Réinitialiser les compteurs
    detection_counts.clear()
    sent_numbers.clear()
    failed_numbers.clear()
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    if not os.path.isdir(IMG_FOLDER):
        print(f"Le dossier '{IMG_FOLDER}' n'existe pas.")
        return

    files = [
        f for f in os.listdir(IMG_FOLDER)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    files.sort()

    if not files:
        print(f"Aucune image trouvée dans '{IMG_FOLDER}'.")
        return

    print(f"Analyse de {len(files)} image(s)...\n")
    
    # Vérifier la configuration Supabase au démarrage
    if not SUPABASE_SERVICE_ROLE_KEY and SUPABASE_KEY:
        print("ATTENTION: Vous utilisez SUPABASE_KEY (clé publique) qui est soumise à RLS.")
        print("   Pour contourner RLS, ajoutez SUPABASE_SERVICE_ROLE_KEY dans votre .env")
        print("   (trouvable dans Supabase > Settings > API > service_role key)\n")

    for filename in files:
        file_path = os.path.join(IMG_FOLDER, filename)
        raw, digits = process_image_file(file_path)
        print(f"--- {filename} ---")
        print("OCR brut:", repr(raw))
        print("Numéro:", digits if digits else "(aucun)")
        
        if digits:
            # Incrémenter le compteur pour ce numéro
            detection_counts[digits] += 1
            count = detection_counts[digits]
            print(f"Occurrences: {count}/{REQUIRED_DETECTIONS}")
            
            # Si le numéro est détecté 3 fois et n'a pas encore été envoyé ou tenté
            if count >= REQUIRED_DETECTIONS and digits not in sent_numbers and digits not in failed_numbers:
                print(f"Numéro {digits} détecté {count} fois - Envoi à Supabase...")
                if save_number_to_supabase(digits):
                    sent_numbers.add(digits)
                    print(f"✅ Numéro {digits} envoyé avec succès à Supabase\n")
                else:
                    failed_numbers.add(digits)
                    print(f"❌ Échec de l'envoi du numéro {digits} (vérifiez votre configuration Supabase)\n")
            elif count >= REQUIRED_DETECTIONS and digits in failed_numbers:
                # Ne pas réessayer si ça a déjà échoué
                print(f"Numéro {digits} déjà tenté (échec précédent) - Ignoré\n")
        else:
            print()
    
    # Résumé final
    print("\n" + "="*50)
    print("RÉSUMÉ")
    print("="*50)
    print(f"Total d'images analysées: {len(files)}")
    print(f"Numéros détectés: {len(detection_counts)}")
    print(f"Numéros envoyés à Supabase: {len(sent_numbers)}")
    print(f"Numéros en échec: {len(failed_numbers)}")
    
    if failed_numbers:
        print("\n ATTENTION: Certains numéros n'ont pas pu être envoyés à Supabase.")
        print("   Vérifiez que SUPABASE_SERVICE_ROLE_KEY est configurée dans votre .env")
        print("   ou créez une politique RLS dans Supabase pour permettre l'INSERT.")
    
    if detection_counts:
        print("\nDétails par numéro:")
        for number, count in sorted(detection_counts.items()):
            if number in sent_numbers:
                status = "✅ Envoyé"
            elif number in failed_numbers:
                status = "❌ Échec"
            else:
                status = f"⏳ {count}/{REQUIRED_DETECTIONS}"
            print(f"  {number}: {count} détection(s) - {status}")


if __name__ == "__main__":
    run_ocr_on_img_folder()