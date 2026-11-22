import os
import random
import cv2
import numpy as np
from config import IMG_SIZE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "dataset_rostros")
COPIAS = 25          # copias degradadas por cada imagen original
SEED = 7
random.seed(SEED)
np.random.seed(SEED)


def es_original_hf(nombre):
    return "_hf_" in nombre or nombre.startswith("desconocido_")


def degradar(img):
    """Imita una foto vista en la pantalla de un celular a traves de la webcam."""
    h, w = img.shape[:2]

    # perdida de resolucion y desenfoque
    f = random.uniform(0.35, 0.8)
    img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.6:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    img = img.astype(np.float32)

    # brillo, contraste y tinte de pantalla
    img = img * random.uniform(0.75, 1.3) + random.uniform(-25, 25)
    img += np.array([random.uniform(-15, 25),
                     random.uniform(-10, 10),
                     random.uniform(-20, 10)])

    # patron moire
    if random.random() < 0.6:
        periodo = random.uniform(3, 8)
        angulo = random.uniform(0, np.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        onda = np.sin((xx * np.cos(angulo) + yy * np.sin(angulo)) * 2 * np.pi / periodo)
        img += onda[..., None] * random.uniform(4, 12)

    # reflejo de luz
    if random.random() < 0.5:
        cx, cy = random.randint(0, w), random.randint(0, h)
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        brillo = np.clip(1 - dist / random.uniform(w * 0.4, w * 0.9), 0, 1)
        img += brillo[..., None] * random.uniform(30, 90)

    # ruido de sensor y compresion JPEG
    img += np.random.normal(0, random.uniform(3, 12), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 75)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main():
    if not os.path.isdir(DATASET):
        raise FileNotFoundError(f"No existe: {DATASET}")

    for persona in sorted(os.listdir(DATASET)):
        carpeta = os.path.join(DATASET, persona)
        if not os.path.isdir(carpeta):
            continue

        # borra copias anteriores para poder repetir el script
        for nombre in os.listdir(carpeta):
            if "_sim_" in nombre:
                os.remove(os.path.join(carpeta, nombre))

        originales = [n for n in os.listdir(carpeta) if es_original_hf(n)]
        total = 0
        for nombre in originales:
            img = cv2.imread(os.path.join(carpeta, nombre))
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            base = os.path.splitext(nombre)[0]
            for k in range(COPIAS):
                cv2.imwrite(os.path.join(carpeta, f"{base}_sim_{k:02d}.jpg"), degradar(img))
                total += 1
        if total:
            print(f"{persona}: {total} copias simuladas")


if __name__ == "__main__":
    main()