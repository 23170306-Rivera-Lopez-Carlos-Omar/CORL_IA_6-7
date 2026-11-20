"""
Tarea 8 - Reconocimiento de rostros en tiempo real.

Flujo:
  1. Lee la webcam con OpenCV.
  2. DETECCION: un detector Haar Cascade encuentra los rostros del cuadro y
     recorta cada uno (region de interes).
  3. INFERENCIA: cada rostro pasa por la CNN (modelo_<MODELO>.keras).
  4. Dibuja la caja con el nombre y el % de confianza. Si la confianza es
     baja muestra "Desconocido".

Requiere haber corrido antes entrenar_cnn.py (con la misma arquitectura).
Controles:  Q o ESC -> salir
"""

import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = "propia"            # "propia" o "mobilenet": igual que ARQUITECTURA en entrenar_cnn.py
TAMANO_IMG = (128, 128)      # el mismo de entrenar_cnn.py
CONFIANZA_MINIMA = 0.70      # por debajo de esto -> "Desconocido"
TAMANO_MIN_CARA = (80, 80)   # ignora detecciones muy pequenas
CAMARA = 0                   # 0 = webcam integrada; prueba 1 si es externa
ESPEJO = True                # voltear como espejo

VERDE = (0, 160, 0)
ROJO = (0, 0, 220)

model = tf.keras.models.load_model(os.path.join(AQUI, f"modelo_{MODELO}.keras"))
with open(os.path.join(AQUI, "clases.json"), encoding="utf-8") as f:
    class_names = json.load(f)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if detector.empty():
    raise SystemExit("No se pudo cargar el detector Haar (usa opencv-python 4.x).")


def preparar_rostro(frame_bgr, x, y, w, h):
    """Recorta y deja el rostro igual que en el entrenamiento.
    - OpenCV da BGR, el modelo se entreno con RGB -> se convierte.
    - Se redimensiona a 128x128.
    - NO se divide entre 255: el modelo ya lo hace (capa Rescaling).
    """
    rostro = cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
    rostro = cv2.resize(rostro, TAMANO_IMG, interpolation=cv2.INTER_AREA)
    return rostro.astype("float32")


def dibujar_etiqueta(frame, x, y, w, h, texto, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y_texto = y - 10 if y - th - 14 > 0 else y + h + th + 10
    cv2.rectangle(frame, (x, y_texto - th - 6), (x + tw + 8, y_texto + base - 2), color, -1)
    cv2.putText(frame, texto, (x + 4, y_texto - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la camara. Revisa que no la use otra app.")

    print(f"Clases: {class_names}\nPresiona Q o ESC para salir.")
    anterior, fps = time.time(), 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if ESPEJO:
                frame = cv2.flip(frame, 1)

            # 1. Deteccion de rostros
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            caras = detector.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5,
                                              minSize=TAMANO_MIN_CARA)

            if len(caras) > 0:
                # 2. Inferencia: todos los rostros en un solo lote
                lote = np.stack([preparar_rostro(frame, *c) for c in caras])
                probabilidades = model(lote, training=False).numpy()   # el modelo ya trae softmax

                # 3. Caja + nombre + confianza
                for (x, y, w, h), probs in zip(caras, probabilidades):
                    i = int(np.argmax(probs))
                    confianza = float(probs[i])
                    if confianza < CONFIANZA_MINIMA:
                        dibujar_etiqueta(frame, x, y, w, h, f"Desconocido ({confianza:.0%})", ROJO)
                    else:
                        dibujar_etiqueta(frame, x, y, w, h, f"{class_names[i]} ({confianza:.0%})", VERDE)

            ahora = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(ahora - anterior, 1e-6))
            anterior = ahora
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Reconocimiento facial - CNN", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
