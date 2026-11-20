from io import BytesIO
import os
import re
import unicodedata

import pandas as pd
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATASET = os.path.join(BASE_DIR, "dataset")
TAMANO_ROSTRO = (128, 128)
MAX_FAMOSOS = 5  # requisito de la tarea: al menos 5 famosos

URL_PARQUET = (
    "hf://datasets/nickmuchi/corise-celeb-identities/"
    "data/train-00000-of-00001-13faf25ef6fe9b6c.parquet"
)


def limpiar_nombre(texto):
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    texto = re.sub(r"[^a-zA-Z0-9]+", "_", texto.strip().lower())
    return texto.strip("_")


def siguiente_numero(carpeta):
    maximo = 0
    if os.path.isdir(carpeta):
        for archivo in os.listdir(carpeta):
            m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", archivo, re.IGNORECASE)
            if m:
                maximo = max(maximo, int(m.group(1)))
    return maximo + 1


def extraer_imagen_pil(valor_imagen):
    if isinstance(valor_imagen, dict):
        if valor_imagen.get("bytes"):
            return Image.open(BytesIO(valor_imagen["bytes"])).convert("RGB")
        if valor_imagen.get("path"):
            return Image.open(valor_imagen["path"]).convert("RGB")
    if isinstance(valor_imagen, (bytes, bytearray)):
        return Image.open(BytesIO(valor_imagen)).convert("RGB")
    raise ValueError("Formato de imagen no reconocido en la columna 'image'.")


def main():
    df = pd.read_parquet(URL_PARQUET)

    famosos_seleccionados = []
    for etiqueta in df["label"]:
        nombre_limpio = limpiar_nombre(etiqueta)
        if nombre_limpio and nombre_limpio not in famosos_seleccionados:
            famosos_seleccionados.append(nombre_limpio)
        if len(famosos_seleccionados) == MAX_FAMOSOS:
            break

    print(f"Famosos seleccionados ({len(famosos_seleccionados)}): {famosos_seleccionados}")

    guardados_total = 0

    for _, fila in df.iterrows():
        famoso = limpiar_nombre(fila["label"])
        if famoso not in famosos_seleccionados:
            continue

        carpeta_salida = os.path.join(CARPETA_DATASET, famoso)
        os.makedirs(carpeta_salida, exist_ok=True)
        numero = siguiente_numero(carpeta_salida)

        img = extraer_imagen_pil(fila["image"])
        img_128 = img.resize(TAMANO_ROSTRO, Image.Resampling.LANCZOS)

        nombre_archivo = f"{famoso}_{numero:04d}.jpg"
        img_128.save(os.path.join(carpeta_salida, nombre_archivo), format="JPEG", quality=95)
        guardados_total += 1

    print(
        f"\nListo. Se guardaron {guardados_total} imágenes de "
        f"{len(famosos_seleccionados)} famosos en: {CARPETA_DATASET}"
    )


if __name__ == "__main__":
    main()