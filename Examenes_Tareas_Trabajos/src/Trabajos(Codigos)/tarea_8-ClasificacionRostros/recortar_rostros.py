import os
import sys
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAMANO = (128, 128)


def recortar_centrado(img_bgr):
    alto, ancho, _ = img_bgr.shape
    lado = min(ancho, alto)
    x1 = (ancho - lado) // 2
    y1 = (alto - lado) // 2
    cuadro = img_bgr[y1 : y1 + lado, x1 : x1 + lado]
    return cv2.resize(cuadro, TAMANO, interpolation=cv2.INTER_AREA)


def main():
    if len(sys.argv) < 3:
        print("Uso: python recortar_rostros.py entrada.jpg salida.jpg")
        return
    entrada = sys.argv[1]
    salida = sys.argv[2]
    img_bgr = cv2.imread(entrada)
    if img_bgr is None:
        print(f"No se pudo abrir: {entrada}")
        return
    resultado = recortar_centrado(img_bgr)
    carpeta_salida = os.path.dirname(os.path.abspath(salida))
    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)
    cv2.imwrite(salida, resultado)
    print(f"Guardado en 128x128: {salida}")


if __name__ == "__main__":
    main()