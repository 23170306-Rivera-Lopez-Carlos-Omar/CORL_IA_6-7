import os
import sys
import time
import cv2
from config import IMG_SIZE, MARGEN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(BASE_DIR, "dataset_rostros")
INTERVALO_AUTO = 0.4     # segundos entre fotos en modo automatico


def main():
    if len(sys.argv) >= 2:
        nombre = sys.argv[1]
    else:
        nombre = input("Nombre de la carpeta (ej. barack_obama): ").strip()
        if not nombre:
            return

    carpeta = os.path.join(SALIDA, nombre)
    os.makedirs(carpeta, exist_ok=True)
    contador = len(os.listdir(carpeta))

    detector = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara.")

    auto = False
    ultimo = 0.0
    print("C = guardar foto | A = modo automatico on/off | Q = salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)             # igual que en_tiempo_real.py
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = detector.detectMultiScale(gris, 1.1, 6, minSize=(60, 60))

        recorte = None
        if len(caras) > 0:
            x, y, w, h = max(caras, key=lambda c: c[2] * c[3])
            m = int(MARGEN * w)
            x1, y1 = max(0, x - m), max(0, y - m)
            x2 = min(frame.shape[1], x + w + m)
            y2 = min(frame.shape[0], y + h + m)
            recorte = cv2.resize(frame[y1:y2, x1:x2], IMG_SIZE,
                                 interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        tecla = cv2.waitKey(1) & 0xFF
        ahora = time.time()
        if tecla == ord("a"):
            auto = not auto
        guardar = tecla == ord("c") or (auto and ahora - ultimo > INTERVALO_AUTO)
        if guardar and recorte is not None:
            contador += 1
            cv2.imwrite(os.path.join(carpeta, f"{nombre}_cam_{contador:04d}.jpg"),
                        recorte)
            ultimo = ahora

        estado = "AUTO" if auto else "MANUAL"
        cv2.putText(frame, f"{nombre} | {estado} | fotos: {contador}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Captura de rostros", frame)
        if tecla == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total en {carpeta}: {contador}")


if __name__ == "__main__":
    main()