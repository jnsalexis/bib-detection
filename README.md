# Bib Detection

Système de détection de numéros de dossards via flux RTSP avec YOLO et OCR.

## Utilisation

### Détection en temps réel
```bash
python stream.py
```

### OCR sur images existantes
```bash
python process_images.py
```

### Interface Web Admin
```bash
python web.py
```

L'interface sera disponible sur `http://localhost:8001` et permet de :
- 📊 Voir tous les numéros détectés dans Supabase
- ⚙️ Éditer les variables d'environnement Supabase
- 🗑️ Supprimer des dossards de la base de données

## Capture du flux vidéo avec ffmpeg

```bash
$ ffmpeg -i "rtsp://admin:teamprod123@192.168.70.101:554/h264Preview_01_main" -vf fps=10 capture_%04d.jpg
```