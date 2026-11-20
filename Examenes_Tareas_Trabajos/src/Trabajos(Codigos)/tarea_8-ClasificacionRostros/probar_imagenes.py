import json
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_RUTA = os.path.join(BASE_DIR, "modelo_rostros.keras")
CLASES_RUTA = os.path.join(BASE_DIR, "clases.json")
IMG_SIZE = (128, 128)


def main():
    if len(sys.argv) < 2:
        print("Uso: python probar_imagenes.py ruta_imagen.jpg")
        return
    ruta_img = sys.argv[1]
    img_bgr = cv2.imread(ruta_img)
    if img_bgr is None:
        print(f"No se pudo leer la imagen: {ruta_img}")
        return
    modelo = tf.keras.models.load_model(MODELO_RUTA)
    with open(CLASES_RUTA, "r", encoding="utf-8") as archivo:
        clases = json.load(archivo)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    redimensionada = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    lote = np.expand_dims(redimensionada.astype("float32") / 255.0, axis=0)
    pred = modelo.predict(lote, verbose=0)[0]
    orden = np.argsort(pred)[::-1]
    idx1 = int(orden[0])
    print(f"Prediccion principal: {clases[idx1]} ({pred[idx1] * 100:.1f}%)")
    for idx in orden:
        i = int(idx)
        print(f"  - {clases[i]}: {pred[i] * 100:.1f}%")


if __name__ == "__main__":
    main()