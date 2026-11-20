from io import BytesIO
import os
import re
import unicodedata
from huggingface_hub import HfApi
import pandas as pd
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATASET = os.path.join(BASE_DIR, "dataset")
TAMANO_ROSTRO = (128, 128)
MAX_FAMOSOS = 5
MAX_POR_FAMOSO = 25
REPO_ID = "pittisl/famous-figures-25"


def limpiar_nombre(texto):
    limpio = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    limpio = re.sub(r"[^a-zA-Z0-9]+", "_", limpio.strip().lower())
    return limpio.strip("_")


def siguiente_numero(carpeta):
    maximo = 0
    if os.path.isdir(carpeta):
        for archivo in os.listdir(carpeta):
            m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", archivo, re.IGNORECASE)
            if m:
                valor = int(m.group(1))
                if valor > maximo:
                    maximo = valor
    return maximo + 1


def extraer_imagen_pil(valor_imagen):
    if isinstance(valor_imagen, Image.Image):
        return valor_imagen.convert("RGB")
    if isinstance(valor_imagen, dict):
        if valor_imagen.get("bytes"):
            return Image.open(BytesIO(valor_imagen["bytes"])).convert("RGB")
        if valor_imagen.get("path"):
            return Image.open(valor_imagen["path"]).convert("RGB")
    if isinstance(valor_imagen, (bytes, bytearray)):
        return Image.open(BytesIO(valor_imagen)).convert("RGB")
    raise ValueError("Formato de imagen no reconocido en la columna 'image'.")


def detectar_columna_etiqueta(df):
    for col in ("name", "label", "person", "identity"):
        if col in df.columns:
            return col
    return df.columns[1]


def obtener_parquets_train(repo_id):
    api = HfApi()
    archivos = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquets = []
    for ruta in archivos:
        if ruta.endswith(".parquet"):
            if "train" in ruta.lower():
                parquets.append(f"hf://datasets/{repo_id}/{ruta}")
    if len(parquets) == 0:
        for ruta in archivos:
            if ruta.endswith(".parquet"):
                parquets.append(f"hf://datasets/{repo_id}/{ruta}")
    return parquets


def main():
    os.makedirs(CARPETA_DATASET, exist_ok=True)
    parquets = obtener_parquets_train(REPO_ID)
    if len(parquets) == 0:
        raise RuntimeError("No se encontraron archivos parquet en el dataset.")
    famosos_seleccionados = []
    conteo_por_famoso = {}
    guardados_total = 0
    for url_pq in parquets:
        df = pd.read_parquet(url_pq)
        col_etiqueta = detectar_columna_etiqueta(df)
        for etiqueta in df[col_etiqueta]:
            nombre_limpio = limpiar_nombre(etiqueta)
            if nombre_limpio:
                if nombre_limpio not in famosos_seleccionados:
                    if len(famosos_seleccionados) < MAX_FAMOSOS:
                        famosos_seleccionados.append(nombre_limpio)
        for _, fila in df.iterrows():
            famoso = limpiar_nombre(fila[col_etiqueta])
            if famoso in famosos_seleccionados:
                actual = conteo_por_famoso.get(famoso, 0)
                if actual < MAX_POR_FAMOSO:
                    carpeta_salida = os.path.join(CARPETA_DATASET, famoso)
                    os.makedirs(carpeta_salida, exist_ok=True)
                    numero = siguiente_numero(carpeta_salida)
                    img = extraer_imagen_pil(fila["image"])
                    img_128 = img.resize(TAMANO_ROSTRO, Image.Resampling.LANCZOS)
                    nombre_archivo = f"{famoso}_{numero:04d}.jpg"
                    img_128.save(
                        os.path.join(carpeta_salida, nombre_archivo),
                        format="JPEG",
                        quality=95,
                    )
                    conteo_por_famoso[famoso] = actual + 1
                    guardados_total += 1
        completo = True
        if len(famosos_seleccionados) < MAX_FAMOSOS:
            completo = False
        for fam in famosos_seleccionados:
            if conteo_por_famoso.get(fam, 0) < MAX_POR_FAMOSO:
                completo = False
        if completo:
            break
    print(f"Famosos seleccionados ({len(famosos_seleccionados)}): {famosos_seleccionados}")
    print("Guardadas por famoso:", conteo_por_famoso)
    print(
        f"\nListo. Se guardaron {guardados_total} imágenes de "
        f"{len(famosos_seleccionados)} famosos en: {CARPETA_DATASET}"
    )


if __name__ == "__main__":
    main()