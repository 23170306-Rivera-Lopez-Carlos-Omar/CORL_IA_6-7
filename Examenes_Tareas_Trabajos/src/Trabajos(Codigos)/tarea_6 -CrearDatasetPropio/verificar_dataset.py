import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATASET = os.path.join(BASE_DIR, "dataset")
TAMANO_ESPERADO = (128, 128)
EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp")


def main():
    if not os.path.isdir(CARPETA_DATASET):
        print(f"No existe la carpeta: {CARPETA_DATASET}")
        return

    subcarpetas = [
        d for d in sorted(os.listdir(CARPETA_DATASET))
        if os.path.isdir(os.path.join(CARPETA_DATASET, d))
    ]

    if not subcarpetas:
        print("No hay subcarpetas dentro de dataset/.")
        return

    total_ok = 0
    problemas = []

    print(f"{'Clase / Carpeta':<28} {'Fotos OK':>10} {'Problemas':>11}")
    print("-" * 52)

    for carpeta in subcarpetas:
        ruta_clase = os.path.join(CARPETA_DATASET, carpeta)
        archivos = sorted(os.listdir(ruta_clase))
        ok_clase = 0
        prob_clase = 0

        for archivo in archivos:
            ruta_img = os.path.join(ruta_clase, archivo)
            if not archivo.lower().endswith(EXTENSIONES):
                prob_clase += 1
                problemas.append(f"{carpeta}/{archivo}: extensión no esperada")
                continue
            try:
                with Image.open(ruta_img) as img:
                    img.verify()
                with Image.open(ruta_img) as img:
                    if img.size != TAMANO_ESPERADO:
                        prob_clase += 1
                        problemas.append(f"{carpeta}/{archivo}: tamaño {img.size} != {TAMANO_ESPERADO}")
                    else:
                        ok_clase += 1
                        total_ok += 1
            except Exception as e:
                prob_clase += 1
                problemas.append(f"{carpeta}/{archivo}: error al abrir ({e})")

        print(f"{carpeta:<28} {ok_clase:>10} {prob_clase:>11}")

    print("-" * 52)
    print(f"Total de carpetas/clases: {len(subcarpetas)}")
    print(f"Total de imágenes válidas 128x128: {total_ok}")

    if problemas:
        print("\nDetalle de problemas encontrados:")
        for p in problemas:
            print("  -", p)
    else:
        print("\nTodo correcto: todas las imágenes están íntegras y en 128x128.")


if __name__ == "__main__":
    main()