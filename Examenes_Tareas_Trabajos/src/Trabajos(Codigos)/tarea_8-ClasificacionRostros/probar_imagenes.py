"""
Tarea 8 - Probar el modelo con fotos sueltas (equivale a las "celdas de prueba"
que dio el maestro, pero para el modelo de rostros).

Uso:
    python probar_imagenes.py                 # usa todas las fotos de la carpeta pruebas/
    python probar_imagenes.py foto1.jpg ...   # o las fotos que indiques

Si la foto trae un rostro, lo recorta con el detector Haar (igual que en
tiempo real). Si no encuentra rostro, usa la foto completa.
"""

import glob
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = "propia"       # "propia" o "mobilenet"
TAMANO_IMG = (128, 128)  # el mismo de entrenar_cnn.py

model = tf.keras.models.load_model(os.path.join(AQUI, f"modelo_{MODELO}.keras"))
with open(os.path.join(AQUI, "clases.json"), encoding="utf-8") as f:
    class_names = json.load(f)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if detector.empty():
    raise SystemExit("No se pudo cargar el detector Haar (usa opencv-python 4.x).")


def preprocesar(ruta):
    """Devuelve (lote, rostro_128, original_rgb).

    lote tiene forma (1, 128, 128, 3) con pixeles de 0 a 255: NO se divide entre
    255 porque el modelo ya lo hace dentro (capa Rescaling).
    """
    bgr = cv2.imread(ruta)
    if bgr is None:
        raise ValueError(f"No se pudo abrir {ruta}")
    original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    caras = detector.detectMultiScale(gris, 1.2, 5, minSize=(60, 60))
    if len(caras) > 0:
        x, y, w, h = max(caras, key=lambda c: c[2] * c[3])   # el rostro mas grande
        recorte = original[y:y + h, x:x + w]
    else:
        recorte = original

    rostro = cv2.resize(recorte, TAMANO_IMG, interpolation=cv2.INTER_AREA)
    lote = np.expand_dims(rostro.astype("float32"), axis=0)
    return lote, rostro, original


def predecir_y_mostrar(ruta):
    lote, rostro, original = preprocesar(ruta)

    probs = model.predict(lote, verbose=0)[0]   # el modelo ya trae softmax
    indice = int(np.argmax(probs))
    clase, confianza = class_names[indice], float(probs[indice])

    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8))
    ejes[0].imshow(original)
    ejes[0].set_title(os.path.basename(ruta))
    ejes[0].axis("off")
    ejes[1].imshow(rostro, interpolation="nearest")
    ejes[1].set_title(f"Asi la ve el modelo (128x128)\n{clase} ({confianza:.1%})")
    ejes[1].axis("off")
    ejes[2].bar(class_names, probs)
    ejes[2].set_ylim(0, 1)
    ejes[2].set_ylabel("Probabilidad")
    ejes[2].set_title("Probabilidad por estudiante")
    ejes[2].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"{os.path.basename(ruta)} -> {clase} (confianza: {confianza:.1%})")


if __name__ == "__main__":
    rutas = sys.argv[1:] or sorted(
        glob.glob(os.path.join(AQUI, "pruebas", "*.jp*g")) + glob.glob(os.path.join(AQUI, "pruebas", "*.png"))
    )
    if not rutas:
        raise SystemExit("No hay fotos. Pon imagenes en la carpeta pruebas/ o pasalas como argumento.")
    for ruta in rutas:
        predecir_y_mostrar(ruta)
