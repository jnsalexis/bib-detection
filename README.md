Lancer la capture du flux vidéo
```
$ ffmpeg -i "rtsp://admin:teamprod123@192.168.70.101:554/h264Preview_01_main" -vf fps=10 capture_%04d.jpg
```