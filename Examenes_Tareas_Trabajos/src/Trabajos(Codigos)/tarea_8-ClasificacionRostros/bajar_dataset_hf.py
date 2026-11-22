import difflib
import os
import random
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPO = "ares1123/celebrity_dataset"

# carpeta local -> nombre en el dataset
FAMOSOS = {
    "angela_merkel": "angela merkel",
    "barack_obama": "barack obama",
    "bill_gates": "bill gates",
    "cristiano_ronaldo": "cristiano ronaldo",
    "denzel_washington": "denzel washington",
    "donald_trump": "donald trump",
    "jackie_chan": "jackie chan",
    "jessica_alba": "jessica alba",
    "lucy_liu": "lucy liu",
    "oprah_winfrey": "oprah winfrey",
}

CANTIDAD_DESCONOCIDOS = 100
MAX_POR_PERSONA = 2      # variedad en "desconocidos"
SEED = 42


def normalizar(t):
    return t.lower().replace("_", " ").strip()


def guardar(img, ruta):
    img.convert("RGB").save(ruta, quality=95)


def main():
    ds = load_dataset(REPO, split="train")
    nombres = ds.features["label"].names
    indice = {normalizar(n): i for i, n in enumerate(nombres)}

    # posiciones de cada persona (sin decodificar imagenes)
    etiquetas = list(ds["label"])
    por_etiqueta = {}
    for pos, e in enumerate(etiquetas):
        por_etiqueta.setdefault(e, []).append(pos)

    # ---------- famosos ----------
    usados = set()
    for carpeta, buscado in FAMOSOS.items():
        clave = normalizar(buscado)
        if clave not in indice:
            parecidos = difflib.get_close_matches(clave, indice.keys(), n=3, cutoff=0.6)
            print(f"[NO ESTA] {buscado} | parecidos: {parecidos}")
            continue
        e = indice[clave]
        usados.add(e)
        destino = os.path.join(DATASET_DIR, carpeta)
        os.makedirs(destino, exist_ok=True)
        posiciones = por_etiqueta[e]
        for k, pos in enumerate(posiciones, start=1):
            guardar(ds[pos]["image"], os.path.join(destino, f"{carpeta}_hf_{k:04d}.jpg"))
        print(f"[OK] {buscado}: {len(posiciones)} imagenes")

    # ---------- desconocidos (nunca incluye a los famosos) ----------
    rng = random.Random(SEED)
    candidatos = [p for p, e in enumerate(etiquetas) if e not in usados]
    rng.shuffle(candidatos)

    elegidos, por_persona = [], {}
    for pos in candidatos:
        e = etiquetas[pos]
        if por_persona.get(e, 0) >= MAX_POR_PERSONA:
            continue
        por_persona[e] = por_persona.get(e, 0) + 1
        elegidos.append(pos)
        if len(elegidos) >= CANTIDAD_DESCONOCIDOS:
            break

    destino = os.path.join(DATASET_DIR, "desconocidos")
    os.makedirs(destino, exist_ok=True)
    for k, pos in enumerate(elegidos, start=1):
        guardar(ds[pos]["image"], os.path.join(destino, f"desconocido_{k:03d}.jpg"))
    print(f"[OK] desconocidos: {len(elegidos)} imagenes de {len(por_persona)} personas")


if __name__ == "__main__":
    main()