import json
import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from config import IMG_SIZE, MARGEN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_RUTA = os.path.join(BASE_DIR, "modelo_rostros.keras")
CLASES_RUTA = os.path.join(BASE_DIR, "clases.json")

CAMARA = 0
UMBRAL_CONFIANZA = 0.55
VENTANA = 5              # predicciones promediadas para estabilidad


def cargar_detector():
    ruta_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(ruta_xml)
    if detector.empty():
        raise RuntimeError("No se pudo inicializar Haar Cascade.")
    return detector


def predecir_rostro(modelo, frame, x, y, w, h):
    """Recorta el rostro con margen y devuelve las probabilidades por clase."""
    alto, ancho = frame.shape[:2]
    m = int(MARGEN * w)
    x1, y1 = max(0, x - m), max(0, y - m)
    x2, y2 = min(ancho, x + w + m), min(alto, y + h + m)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    lote = np.expand_dims(rgb.astype("float32"), axis=0)   # el modelo normaliza solo
    return modelo.predict(lote, verbose=0)[0]


def dibujar_etiqueta(frame, x, y, w, h, texto1, texto2, color):
    alto, ancho = frame.shape[:2]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    pos_x = max(5, x)
    pos_y = y - 42
    if pos_y < 5:
        pos_y = min(y + h + 8, alto - 48)
    box_x2 = min(ancho - 5, pos_x + 345)

    cv2.rectangle(frame, (pos_x, pos_y), (box_x2, pos_y + 43), (20, 20, 20), -1)
    cv2.putText(frame, texto1, (pos_x + 5, pos_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
    cv2.putText(frame, texto2, (pos_x + 5, pos_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


def main():
    if not os.path.isfile(MODELO_RUTA) or not os.path.isfile(CLASES_RUTA):
        raise FileNotFoundError("Ejecuta primero entrenar_cnn.py")

    modelo = tf.keras.models.load_model(MODELO_RUTA)
    with open(CLASES_RUTA, "r", encoding="utf-8") as f:
        clases = json.load(f)
    if modelo.output_shape[-1] != len(clases):
        raise ValueError("El numero de clases no coincide con el modelo.")
    print("Clases cargadas:", clases)

    detector = cargar_detector()
    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara web.")
    print("Presiona Q o ESC para salir.")

    historial = deque(maxlen=VENTANA)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detecciones = detector.detectMultiScale(
                gris, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

            cv2.putText(frame, f"Rostros detectados: {len(detecciones)}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            if len(detecciones) == 0:
                historial.clear()
                cv2.putText(frame, "No se detecta ningun rostro", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            for (x, y, w, h) in detecciones:
                probs = predecir_rostro(modelo, frame, x, y, w, h)
                if probs is None:
                    continue

                # suavizado temporal solo con un rostro en cuadro
                if len(detecciones) == 1:
                    historial.append(probs)
                    probs = np.mean(historial, axis=0)
                else:
                    historial.clear()

                orden = np.argsort(probs)[::-1]
                i1 = int(orden[0])
                i2 = int(orden[1]) if len(orden) > 1 else i1
                nombre, conf = clases[i1], float(probs[i1])
                segunda, conf2 = clases[i2], float(probs[i2])

                if nombre == "desconocidos" or conf < UMBRAL_CONFIANZA:
                    texto1 = f"Desconocido: {conf * 100:.1f}%"
                    color = (0, 165, 255)
                else:
                    texto1 = f"{nombre}: {conf * 100:.1f}%"
                    color = (0, 255, 0)
                texto2 = f"2da: {segunda} ({conf2 * 100:.1f}%)"

                dibujar_etiqueta(frame, x, y, w, h, texto1, texto2, color)

            cv2.imshow("Reconocimiento Facial CNN - Tarea 8", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camara cerrada.")


if __name__ == "__main__":
    main()