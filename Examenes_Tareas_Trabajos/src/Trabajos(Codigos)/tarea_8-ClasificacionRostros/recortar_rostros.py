"""
Tarea 8 (OPCIONAL) - Recorta los rostros de fotos completas para armar dataset/.

Solo necesitas esto si las fotos del Dataset (#6) NO estan ya recortadas a la cara.

Entrada:  fotos_originales/<persona>/*.jpg
Salida:   dataset/<persona>/*.jpg   (solo el rostro, 128x128)

Uso:  python recortar_rostros.py
"""

import os

import cv2

AQUI = os.path.dirname(os.path.abspath(__file__))
ENTRADA = os.path.join(AQUI, "fotos_originales")
SALIDA = os.path.join(AQUI, "dataset")
EXTENSIONES = (".jpg", ".jpeg", ".png")

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if detector.empty():
    raise SystemExit("No se pudo cargar el detector Haar (usa opencv-python 4.x).")

for persona in sorted(os.listdir(ENTRADA)):
    carpeta = os.path.join(ENTRADA, persona)
    if not os.path.isdir(carpeta):
        continue
    os.makedirs(os.path.join(SALIDA, persona), exist_ok=True)
    guardadas = descartadas = 0

    for nombre in sorted(os.listdir(carpeta)):
        if not nombre.lower().endswith(EXTENSIONES):
            continue
        img = cv2.imread(os.path.join(carpeta, nombre))
        if img is None:
            descartadas += 1
            continue
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        caras = detector.detectMultiScale(gris, 1.2, 5, minSize=(60, 60))
        if len(caras) == 0:
            descartadas += 1   # sin rostro detectado: se omite
            continue
        x, y, w, h = max(caras, key=lambda c: c[2] * c[3])   # el rostro mas grande
        rostro = cv2.resize(img[y:y + h, x:x + w], (128, 128), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(SALIDA, persona, nombre), rostro)
        guardadas += 1

    print(f"{persona:<15} guardadas: {guardadas:>4} | sin rostro/ilegibles: {descartadas}")
