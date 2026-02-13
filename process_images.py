import os
import re
import time
import json
import cv2
import numpy as np
from PIL import Image
import pytesseract
from collections import defaultdict
from typing import Optional, Set
from supabase import create_client, Client
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

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

# Dossier contenant les images préprocessées à analyser
IMG_FOLDER = config["folders"]["output_folder"]
# Dossier pour déplacer les images traitées
IMG_PROCESSED_FOLDER = config["folders"]["processed_folder"]
# Nombre de détections requises pour envoyer à Supabase
REQUIRED_DETECTIONS = config["detection"]["required_detections"]

# Créer les dossiers si nécessaires
os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(IMG_PROCESSED_FOLDER, exist_ok=True)

# Configurations Tesseract pour les dossards (chiffres uniquement)
CONFIG_PSM8 = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
CONFIG_PSM7 = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
CONFIG_PSM6 = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
CONFIG_PSM13 = "--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789"  # Texte brut, ligne simple

# Configuration Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# Utiliser SUPABASE_SERVICE_ROLE_KEY si disponible (contourne RLS), sinon SUPABASE_KEY
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "dossards")

# Client Supabase (initialisé à None, sera créé à la première utilisation)
_supabase_client: Optional[Client] = None

# Compteur des détections par numéro
detection_counts = defaultdict(int)
# Numéros déjà envoyés à Supabase (pour éviter les doublons)
sent_numbers = set()
# Numéros pour lesquels l'envoi a échoué (pour éviter de réessayer indéfiniment)
failed_numbers = set()

print("="*60)
print("CONFIGURATION CHARGÉE")
print("="*60)
print(f"Dossier à surveiller: {IMG_FOLDER}")
print(f"Dossier de sortie: {IMG_PROCESSED_FOLDER}")
print(f"Détections requises: {REQUIRED_DETECTIONS}")
print("="*60 + "\n")


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


def is_valid_bib_number(digits):
    """
    Valide qu'un numéro ressemble à un dossard de course.
    """
    if not digits:
        return False
    
    # Trop court ou trop long
    if len(digits) < 1 or len(digits) > 6:
        return False
    
    # Pas d'année (4 chiffres commençant par 20)
    if len(digits) == 4 and digits.startswith("20"):
        return False
    
    # Pas que des zéros
    if digits == "0" * len(digits):
        return False
    
    # Pas que des 1
    if digits == "1" * len(digits) and len(digits) > 3:
        return False
    
    return True


def score_ocr_result(digits):
    """
    Score un résultat OCR pour sélectionner le meilleur.
    Plus le score est élevé, meilleur est le résultat.
    """
    if not is_valid_bib_number(digits):
        return 0
    
    score = len(digits) * 10
    
    # Bonus pour les numéros de 3-5 chiffres (dossards typiques)
    if 3 <= len(digits) <= 5:
        score += 20
    
    # Pénaliser les numéros très courts
    if len(digits) == 1:
        score -= 15
    
    # Pénaliser les numéros très longs
    if len(digits) > 5:
        score -= 10
    
    return score


def ocr_image(img_binary):
    """
    Lance l'OCR sur une image binarisée avec plusieurs PSM et retourne
    le meilleur résultat (texte brut et numéro extrait).
    
    Args:
        img_binary: Image binarisée (numpy array)
    
    Returns:
        Tuple (raw_text, digits)
    """
    pil = Image.fromarray(img_binary)
    results = []

    for config in (CONFIG_PSM8, CONFIG_PSM7, CONFIG_PSM6, CONFIG_PSM13):
        try:
            raw = pytesseract.image_to_string(pil, config=config).strip()
            digits = "".join(c for c in raw if c.isdigit())
            
            if digits and is_valid_bib_number(digits):
                score = score_ocr_result(digits)
                results.append((raw, digits, score))
        except Exception:
            continue

    if not results:
        return ("", "")

    # Prendre le résultat avec le meilleur score
    best = max(results, key=lambda x: x[2])
    return (best[0], best[1])


def process_image_file(image_path):
    """
    Charge une image déjà préprocessée et effectue l'OCR.
    L'image doit être déjà binarisée et prête pour l'OCR.
    
    Returns:
        (raw_text, numbers_str)
    """
    try:
        # Lire l'image préprocessée (déjà en niveaux de gris/binarisée)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ("", "")

        raw, digits = ocr_image(img)
        return (raw, digits)
    except Exception as e:
        print(f"  Erreur lors du traitement: {e}")
        return ("", "")


def run_ocr_watcher():
    """
    Surveille en continu le dossier img/ et effectue l'OCR sur les nouvelles images.
    Les images sont déplacées vers img_processed/ après traitement.
    Compte les occurrences de chaque numéro et envoie à Supabase quand un numéro est détecté 3 fois.
    """
    global detection_counts, sent_numbers, failed_numbers
    
    # Réinitialiser les compteurs
    detection_counts.clear()
    sent_numbers.clear()
    failed_numbers.clear()
    
    # Ensemble des fichiers déjà traités (pour éviter les doublons)
    processed_files: Set[str] = set()
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    if not os.path.isdir(IMG_FOLDER):
        print(f"Le dossier '{IMG_FOLDER}' n'existe pas. Création...")
        os.makedirs(IMG_FOLDER, exist_ok=True)
    
    print("="*60)
    print("OCR WATCHER - Surveillance en continu du dossier img/")
    print("="*60)
    
    # Vérifier la configuration Supabase au démarrage
    if not SUPABASE_SERVICE_ROLE_KEY and SUPABASE_KEY:
        print("\n⚠️  ATTENTION: Vous utilisez SUPABASE_KEY (clé publique) qui est soumise à RLS.")
        print("   Pour contourner RLS, ajoutez SUPABASE_SERVICE_ROLE_KEY dans votre .env")
        print("   (trouvable dans Supabase > Settings > API > service_role key)\n")
    
    print(f"En attente de nouvelles images dans '{IMG_FOLDER}'...")
    print("Appuyez sur Ctrl+C pour arrêter.\n")
    
    try:
        while True:
            # Scanner le dossier pour de nouvelles images
            current_files = {
                f for f in os.listdir(IMG_FOLDER)
                if os.path.splitext(f.lower())[1] in image_extensions
            }
            
            # Trouver les nouveaux fichiers
            new_files = current_files - processed_files
            
            for filename in sorted(new_files):
                file_path = os.path.join(IMG_FOLDER, filename)
                
                # Vérifier que le fichier existe toujours (peut avoir été supprimé)
                if not os.path.exists(file_path):
                    processed_files.add(filename)
                    continue
                
                # Attendre un peu pour être sûr que l'écriture est terminée
                time.sleep(0.1)
                
                print(f"📸 Nouvelle image: {filename}")
                
                # Effectuer l'OCR
                raw, digits = process_image_file(file_path)
                print(f"   OCR brut: {repr(raw)}")
                print(f"   Numéro: {digits if digits else '(aucun)'}")
                
                if digits:
                    # Incrémenter le compteur pour ce numéro
                    detection_counts[digits] += 1
                    count = detection_counts[digits]
                    print(f"   Occurrences: {count}/{REQUIRED_DETECTIONS}")
                    
                    # Si le numéro est détecté 3 fois et n'a pas encore été envoyé ou tenté
                    if count >= REQUIRED_DETECTIONS and digits not in sent_numbers and digits not in failed_numbers:
                        print(f"   🚀 Numéro {digits} détecté {count} fois - Envoi à Supabase...")
                        if save_number_to_supabase(digits):
                            sent_numbers.add(digits)
                            print(f"   ✅ Numéro {digits} envoyé avec succès à Supabase")
                        else:
                            failed_numbers.add(digits)
                            print(f"   ❌ Échec de l'envoi du numéro {digits}")
                    elif count >= REQUIRED_DETECTIONS and digits in sent_numbers:
                        print(f"   ✅ Numéro {digits} déjà envoyé à Supabase")
                    elif count >= REQUIRED_DETECTIONS and digits in failed_numbers:
                        print(f"   ⚠️  Numéro {digits} déjà tenté (échec précédent)")
                
                # Déplacer l'image vers img_processed
                try:
                    dest_path = os.path.join(IMG_PROCESSED_FOLDER, filename)
                    os.rename(file_path, dest_path)
                    print(f"   ➡️  Déplacé vers {IMG_PROCESSED_FOLDER}/")
                except Exception as e:
                    print(f"   ⚠️  Impossible de déplacer le fichier: {e}")
                
                print()  # Ligne vide pour lisibilité
                
                # Marquer comme traité
                processed_files.add(filename)
            
            # Attendre avant le prochain scan (éviter de surcharger le CPU)
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("ARRÊT DU WATCHER - RÉSUMÉ")
        print("="*60)
        print(f"Total d'images analysées: {len(processed_files)}")
        print(f"Numéros détectés: {len(detection_counts)}")
        print(f"Numéros envoyés à Supabase: {len(sent_numbers)}")
        print(f"Numéros en échec: {len(failed_numbers)}")
        
        if failed_numbers:
            print("\n⚠️  ATTENTION: Certains numéros n'ont pas pu être envoyés à Supabase.")
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
        
        print("\nAu revoir! 👋")


if __name__ == "__main__":
    run_ocr_watcher()