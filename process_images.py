#=================================================================================
# IMPORTS
#=================================================================================
import os
import time
import json
import cv2
from PIL import Image
import pytesseract
from collections import defaultdict
from typing import Optional, Set
from supabase import create_client, Client
from dotenv import load_dotenv


#=================================================================================
# CONFIGURATION
#=================================================================================
load_dotenv()


def load_config():
    """Charge la configuration depuis config.json."""
    if not os.path.exists("config.json"):
        raise FileNotFoundError("Fichier config.json introuvable.")
    
    with open("config.json", "r") as f:
        return json.load(f)


config = load_config()

IMG_FOLDER = config["folders"]["output_folder"]
IMG_PROCESSED_FOLDER = config["folders"]["processed_folder"]
REQUIRED_DETECTIONS = config["detection"]["required_detections"]

os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(IMG_PROCESSED_FOLDER, exist_ok=True)

CONFIG_PSM8 = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
CONFIG_PSM7 = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
CONFIG_PSM6 = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
CONFIG_PSM13 = "--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789"

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "dossards")

print("="*60)
print("CONFIGURATION")
print("="*60)
print(f"Dossier a surveiller: {IMG_FOLDER}")
print(f"Dossier de sortie: {IMG_PROCESSED_FOLDER}")
print(f"Detections requises: {REQUIRED_DETECTIONS}")
print("="*60 + "\n")


#=================================================================================
# VARIABLES GLOBALES
#=================================================================================
_supabase_client: Optional[Client] = None
detection_counts = defaultdict(int)
sent_numbers = set()
failed_numbers = set()


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


def save_number_to_supabase(number: str) -> bool:
    """Enregistre un numero dans Supabase."""
    client = get_supabase_client()
    
    if client is None:
        print("Supabase non configure")
        return False
    
    try:
        number_int = int(number)
        
        existing = client.table(SUPABASE_TABLE).select("id").eq("number", number_int).execute()
        
        if existing.data:
            print(f"Numero {number_int} existe deja dans Supabase")
            return True
        else:
            result = client.table(SUPABASE_TABLE).insert({"number": number_int}).execute()
            print(f"✅ Numero {number_int} enregistre dans Supabase")
            return True
        
    except ValueError:
        print(f"❌ Erreur: '{number}' n'est pas un numero valide")
        return False
    except Exception as e:
        error_msg = str(e)
        if "row-level security policy" in error_msg.lower() or "42501" in error_msg:
            print(f"❌ Erreur RLS: insertion bloquee par les politiques de securite")
        else:
            print(f"❌ Erreur lors de l'enregistrement: {e}")
        return False


#=================================================================================
# FONCTIONS - OCR
#=================================================================================
def is_valid_bib_number(digits):
    """Valide qu'un numero ressemble a un dossard."""
    if not digits:
        return False
    
    if len(digits) < 1 or len(digits) > 6:
        return False
    
    if len(digits) == 4 and digits.startswith("20"):
        return False
    
    if digits == "0" * len(digits):
        return False
    
    if digits == "1" * len(digits) and len(digits) > 3:
        return False
    
    return True


def score_ocr_result(digits):
    """Score un resultat OCR."""
    if not is_valid_bib_number(digits):
        return 0
    
    score = len(digits) * 10
    
    if 3 <= len(digits) <= 5:
        score += 20
    
    if len(digits) == 1:
        score -= 15
    
    if len(digits) > 5:
        score -= 10
    
    return score


def ocr_image(img_binary):
    """Lance l'OCR sur une image binarisee."""
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

    best = max(results, key=lambda x: x[2])
    return (best[0], best[1])


def process_image_file(image_path):
    """Charge une image et effectue l'OCR."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ("", "")

        raw, digits = ocr_image(img)
        return (raw, digits)
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        return ("", "")


#=================================================================================
# FONCTION PRINCIPALE
#=================================================================================
def run_ocr_watcher():
    """Surveille le dossier img/ et effectue l'OCR."""
    global detection_counts, sent_numbers, failed_numbers
    
    detection_counts.clear()
    sent_numbers.clear()
    failed_numbers.clear()
    
    processed_files: Set[str] = set()
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    if not os.path.isdir(IMG_FOLDER):
        print(f"Dossier '{IMG_FOLDER}' introuvable. Creation...")
        os.makedirs(IMG_FOLDER, exist_ok=True)
    
    print("="*60)
    print("OCR WATCHER - Surveillance en continu")
    print("="*60)
    
    if not SUPABASE_SERVICE_ROLE_KEY and SUPABASE_KEY:
        print("ATTENTION: Vous utilisez SUPABASE_KEY (cle publique)")
        print("Pour contourner RLS, ajoutez SUPABASE_SERVICE_ROLE_KEY dans .env\n")
    
    print(f"En attente de nouvelles images dans '{IMG_FOLDER}'...")
    print("Appuyez sur Ctrl+C pour arreter.\n")
    
    try:
        while True:
            current_files = {
                f for f in os.listdir(IMG_FOLDER)
                if os.path.splitext(f.lower())[1] in image_extensions
            }
            
            new_files = current_files - processed_files
            
            for filename in sorted(new_files):
                file_path = os.path.join(IMG_FOLDER, filename)
                
                if not os.path.exists(file_path):
                    processed_files.add(filename)
                    continue
                
                time.sleep(0.1)
                
                print(f"Nouvelle image: {filename}")
                
                raw, digits = process_image_file(file_path)
                print(f"  OCR brut: {repr(raw)}")
                print(f"  Numero: {digits if digits else '(aucun)'}")
                
                if digits:
                    detection_counts[digits] += 1
                    count = detection_counts[digits]
                    print(f"  Occurrences: {count}/{REQUIRED_DETECTIONS}")
                    
                    if count >= REQUIRED_DETECTIONS and digits not in sent_numbers and digits not in failed_numbers:
                        print(f"  Numero {digits} detecte {count} fois - Envoi a Supabase...")
                        if save_number_to_supabase(digits):
                            sent_numbers.add(digits)
                        else:
                            failed_numbers.add(digits)
                    elif count >= REQUIRED_DETECTIONS and digits in sent_numbers:
                        print(f"  ✅ Numero {digits} deja envoye a Supabase")
                    elif count >= REQUIRED_DETECTIONS and digits in failed_numbers:
                        print(f"  Numero {digits} deja tente (echec precedent)")
                
                try:
                    dest_path = os.path.join(IMG_PROCESSED_FOLDER, filename)
                    os.rename(file_path, dest_path)
                    print(f"  Deplace vers {IMG_PROCESSED_FOLDER}/")
                except Exception as e:
                    print(f"  Impossible de deplacer le fichier: {e}")
                
                print()
                processed_files.add(filename)
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("ARRET DU WATCHER - RESUME")
        print("="*60)
        print(f"Total d'images analysees: {len(processed_files)}")
        print(f"Numeros detectes: {len(detection_counts)}")
        print(f"Numeros envoyes a Supabase: {len(sent_numbers)}")
        print(f"Numeros en echec: {len(failed_numbers)}")
        
        if failed_numbers:
            print("\nATTENTION: Certains numeros n'ont pas pu etre envoyes a Supabase.")
        
        if detection_counts:
            print("\nDetails par numero:")
            for number, count in sorted(detection_counts.items()):
                if number in sent_numbers:
                    status = "✅ Envoye"
                elif number in failed_numbers:
                    status = "❌ Echec"
                else:
                    status = f"{count}/{REQUIRED_DETECTIONS}"
                print(f"  {number}: {count} detection(s) - {status}")


#=================================================================================
# MAIN
#=================================================================================
if __name__ == "__main__":
    run_ocr_watcher()
