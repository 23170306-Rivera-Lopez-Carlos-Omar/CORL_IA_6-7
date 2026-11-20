from io import BytesIO
import os
import re

import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA = os.path.join(BASE_DIR, "dataset", "rivera_lopez_carlos_omar")
PREFIJO = "rivera_lopez_carlos_omar"
TAMANO = (128, 128)
CAMARA = 0  # cambia a 1 si usas cámara externa


def siguiente_numero(carpeta):
    maximo = 0
    if os.path.isdir(carpeta):
        for archivo in os.listdir(carpeta):
            m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", archivo, re.IGNORECASE)
            if m:
                maximo = max(maximo, int(m.group(1)))
    return maximo + 1


def recortar_cuadro_centrado(img):
    ancho, alto = img.size
    lado = min(ancho, alto)
    izq = (ancho - lado) // 2
    arriba = (alto - lado) // 2
    return img.crop((izq, arriba, izq + lado, arriba + lado))


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara. Revisa que no la use otra app.")

    numero = siguiente_numero(CARPETA_SALIDA)
    guardadas = 0

    print("Controles:")
    print("  [ESPACIO] -> Tomar y guardar foto en 128x128")
    print("  [Q] o [ESC] -> Salir")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer cuadro de la cámara.")
                break

            espejo = cv2.flip(frame, 1)
            alto, ancho, _ = espejo.shape
            lado = min(ancho, alto)
            x1 = (ancho - lado) // 2
            y1 = (alto - lado) // 2

            vista = espejo.copy()
            cv2.rectangle(vista, (x1, y1), (x1 + lado, y1 + lado), (0, 255, 0), 2)
            cv2.putText(
                vista,
                f"Guardadas: {guardadas} | ESPACIO = foto | Q = salir",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Capturar fotos 128x128", vista)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord(" "):
                ok_enc, buf = cv2.imencode(".png", espejo)
                if ok_enc:
                    img_pil = Image.open(BytesIO(buf.tobytes())).convert("RGB")
                    img_128 = recortar_cuadro_centrado(img_pil).resize(
                        TAMANO, Image.Resampling.LANCZOS
                    )
                    nombre = f"{PREFIJO}_{numero:04d}.jpg"
                    img_128.save(
                        os.path.join(CARPETA_SALIDA, nombre), format="JPEG", quality=95
                    )
                    numero += 1
                    guardadas += 1
                    print(f"  -> Guardada {nombre}")
            elif tecla in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Listo: {guardadas} fotos nuevas en '{CARPETA_SALIDA}'.")


if __name__ == "__main__":
    main()