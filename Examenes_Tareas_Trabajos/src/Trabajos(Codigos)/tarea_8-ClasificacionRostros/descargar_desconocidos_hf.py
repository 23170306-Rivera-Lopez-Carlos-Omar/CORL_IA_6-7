import io
import os
from datasets import load_dataset
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALIDA_DIR = os.path.join(BASE_DIR, "dataset", "desconocidos")
TAMANO = (128, 128)
CANTIDAD = 300
DATASET_PRINCIPAL = "tonyassi/celebrity-1000"
DATASET_RESPALDO = "sammyboi1801/lfw-face-transformer-dataset"


def limpiar_carpeta(carpeta):
    if os.path.isdir(carpeta):
        for nombre in os.listdir(carpeta):
            ruta = os.path.join(carpeta, nombre)
            if os.path.isfile(ruta):
                os.remove(ruta)


def extraer_imagen(fila):
    for _, valor in fila.items():
        if isinstance(valor, Image.Image):
            return valor.convert("RGB")
        if isinstance(valor, dict):
            datos_bytes = valor.get("bytes")
            if datos_bytes is not None:
                return Image.open(io.BytesIO(datos_bytes)).convert("RGB")
            ruta_img = valor.get("path")
            if ruta_img is not None:
                if os.path.isfile(ruta_img):
                    return Image.open(ruta_img).convert("RGB")
        if isinstance(valor, bytes):
            return Image.open(io.BytesIO(valor)).convert("RGB")
    return None


def recortar_y_guardar(img, ruta_destino):
    ancho, alto = img.size
    lado = min(ancho, alto)
    izq = (ancho - lado) // 2
    sup = (alto - lado) // 2
    recorte = img.crop((izq, sup, izq + lado, sup + lado))
    final = recorte.resize(TAMANO, Image.Resampling.LANCZOS)
    final.save(ruta_destino, format="JPEG", quality=92)


def abrir_stream():
    for split_nombre in ("train", "validation", "test"):
        try:
            ds = load_dataset(DATASET_PRINCIPAL, split=split_nombre, streaming=True)
            return ds
        except Exception:
            pass
    return load_dataset(DATASET_RESPALDO, split="train", streaming=True)


def main():
    os.makedirs(SALIDA_DIR, exist_ok=True)
    limpiar_carpeta(SALIDA_DIR)
    ds = abrir_stream()
    guardadas = 0
    for fila in ds:
        img = extraer_imagen(fila)
        if img is not None:
            destino = os.path.join(SALIDA_DIR, f"famoso_{guardadas + 1:03d}.jpg")
            recortar_y_guardar(img, destino)
            guardadas += 1
            print(f"[OK] {destino}")
            if guardadas >= CANTIDAD:
                break
    print(f"Total guardadas en {SALIDA_DIR}: {guardadas}")


if __name__ == "__main__":
    main()