import os
import cv2
from config import IMG_SIZE, MARGEN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENTRADA = os.path.join(BASE_DIR, "dataset")
SALIDA = os.path.join(BASE_DIR, "dataset_rostros")
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp")

detector = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)


def es_de_huggingface(nombre):
    # estas fotos ya vienen recortadas en la cara
    return "_hf_" in nombre or nombre.startswith("desconocido_")


def detectar_cara(img):
    """Devuelve (x, y, w, h) de la cara mas grande, o None."""
    alto, ancho = img.shape[:2]
    p = int(0.4 * max(alto, ancho))    # borde para caras que llenan la foto
    ampliada = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REPLICATE)
    gris = cv2.cvtColor(ampliada, cv2.COLOR_BGR2GRAY)
    caras = detector.detectMultiScale(gris, 1.1, 5, minSize=(40, 40))
    if len(caras) == 0:
        return None
    x, y, w, h = max(caras, key=lambda c: c[2] * c[3])
    return x - p, y - p, w, h


def recorte_con_margen(img, x, y, w, h):
    alto, ancho = img.shape[:2]
    m = int(MARGEN * w)
    x1, y1 = max(0, x - m), max(0, y - m)
    x2, y2 = min(ancho, x + w + m), min(alto, y + h + m)
    return img[y1:y2, x1:x2]


def recorte_centro(img):
    alto, ancho = img.shape[:2]
    lado = int(min(alto, ancho) * 0.85)
    x1, y1 = (ancho - lado) // 2, (alto - lado) // 2
    return img[y1:y1 + lado, x1:x1 + lado]


def main():
    if not os.path.isdir(ENTRADA):
        raise FileNotFoundError(f"No existe: {ENTRADA}")

    for persona in sorted(os.listdir(ENTRADA)):
        carpeta = os.path.join(ENTRADA, persona)
        if not os.path.isdir(carpeta):
            continue
        destino = os.path.join(SALIDA, persona)
        os.makedirs(destino, exist_ok=True)

        ok = centro = descartadas = 0
        for nombre in sorted(os.listdir(carpeta)):
            if not nombre.lower().endswith(EXTENSIONES):
                continue
            img = cv2.imread(os.path.join(carpeta, nombre))
            if img is None:
                continue

            caja = detectar_cara(img)
            if caja is not None:
                recorte = recorte_con_margen(img, *caja)
                ok += 1
            elif es_de_huggingface(nombre):
                recorte = recorte_centro(img)
                centro += 1
            else:
                descartadas += 1       # foto propia sin cara detectada
                continue

            if recorte.size == 0:
                descartadas += 1
                continue
            salida = cv2.resize(recorte, IMG_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(destino, nombre), salida)

        print(f"{persona}: {ok} con Haar, {centro} recorte central, {descartadas} descartadas")


if __name__ == "__main__":
    main()  