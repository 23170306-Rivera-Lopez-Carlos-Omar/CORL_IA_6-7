import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from config import IMG_SIZE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "dataset_rostros")
MODELO_SALIDA = os.path.join(BASE_DIR, "modelo_rostros.keras")
CLASES_SALIDA = os.path.join(BASE_DIR, "clases.json")
GRAFICA_SALIDA = os.path.join(BASE_DIR, "curvas_entrenamiento.png")

BATCH_SIZE = 16
EPOCHS_CABEZA = 30      # fase 1: base congelada
EPOCHS_FINE = 30        # fase 2: fine-tuning
SEED = 42
MIN_IMAGENES = 30       # aviso si una clase tiene menos


def revisar_dataset():
    """Cuenta imagenes por persona y avisa de clases vacias o pequenas."""
    if not os.path.isdir(DATASET):
        raise FileNotFoundError(f"No existe: {DATASET}")

    extensiones = (".jpg", ".jpeg", ".png", ".bmp")
    conteo = {}
    for persona in sorted(os.listdir(DATASET)):
        carpeta = os.path.join(DATASET, persona)
        if os.path.isdir(carpeta):
            conteo[persona] = sum(
                1 for f in os.listdir(carpeta) if f.lower().endswith(extensiones))

    print("\nImagenes por persona en dataset_rostros:")
    for persona, n in conteo.items():
        aviso = ""
        if n == 0:
            aviso = "  <-- VACIA"
        elif n < MIN_IMAGENES:
            aviso = f"  <-- pocas (menos de {MIN_IMAGENES})"
        print(f"  {persona}: {n}{aviso}")

    if any(n == 0 for n in conteo.values()):
        raise ValueError("Hay carpetas vacias en dataset_rostros.")


def cargar_datasets():
    """Division 80% entrenamiento / 20% validacion."""
    args = dict(labels="inferred", label_mode="categorical", image_size=IMG_SIZE,
                batch_size=BATCH_SIZE, seed=SEED, validation_split=0.2)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET, subset="training", shuffle=True, **args)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET, subset="validation", shuffle=False, **args)

    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise ValueError("Se necesitan al menos dos personas diferentes.")

    with open(CLASES_SALIDA, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    # pesos de clase para compensar desbalance
    conteos = np.zeros(len(class_names), dtype=np.float32)
    for _, etiquetas in train_ds.unbatch():
        conteos[int(np.argmax(etiquetas.numpy()))] += 1
    total, n = conteos.sum(), len(class_names)
    class_weight = {
        i: float(min(total / (n * conteos[i]), 4.0)) if conteos[i] > 0 else 1.0
        for i in range(n)
    }

    print("\nPersonas:", class_names)
    print("Imagenes por clase (train):",
          dict(zip(class_names, conteos.astype(int).tolist())))

    auto = tf.data.AUTOTUNE
    return train_ds.cache().prefetch(auto), val_ds.cache().prefetch(auto), class_names, class_weight


def construir_modelo(num_clases):
    """MobileNetV2 (transfer learning) + cabeza con BatchNorm y Dropout."""
    aumento = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.12),
        layers.RandomBrightness(0.3),
        layers.RandomContrast(0.3),
    ], name="data_augmentation")

    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    entradas = layers.Input(shape=IMG_SIZE + (3,))
    x = aumento(entradas)
    # normalizacion a [-1, 1]; va dentro del modelo, por eso la camara envia 0-255
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalizacion")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    salidas = layers.Dense(num_clases, activation="softmax")(x)

    return models.Model(entradas, salidas, name="cnn_rostros_mobilenetv2"), base


def compilar(modelo, lr):
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                   loss="categorical_crossentropy", metrics=["accuracy"])


def guardar_curvas(historias):
    h = {k: sum((hist.history[k] for hist in historias), [])
         for k in ("accuracy", "val_accuracy", "loss", "val_loss")}
    ep = range(1, len(h["accuracy"]) + 1)
    corte = len(historias[0].history["accuracy"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tr, va, titulo in (
        (a1, "accuracy", "val_accuracy", "Precision (Accuracy)"),
        (a2, "loss", "val_loss", "Perdida (Loss)"),
    ):
        ax.plot(ep, h[tr], label="Entrenamiento")
        ax.plot(ep, h[va], label="Validacion")
        ax.axvline(corte, color="gray", linestyle="--", label="Inicio fine-tuning")
        ax.set_title(titulo)
        ax.set_xlabel("Epocas")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(GRAFICA_SALIDA, dpi=150)
    plt.close(fig)
    print("Grafica guardada:", GRAFICA_SALIDA)


def evaluar_por_clase(val_ds, class_names):
    """Accuracy de validacion por persona usando el mejor modelo guardado."""
    modelo = tf.keras.models.load_model(MODELO_SALIDA)
    y_true, y_pred = [], []
    for lote, etiquetas in val_ds:
        probs = modelo.predict(lote, verbose=0)
        y_true += np.argmax(etiquetas.numpy(), axis=1).tolist()
        y_pred += np.argmax(probs, axis=1).tolist()
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    print("\n=== ACCURACY DE VALIDACION POR PERSONA ===")
    for i, nombre in enumerate(class_names):
        mascara = y_true == i
        if mascara.sum() == 0:
            continue
        acc = (y_pred[mascara] == i).mean() * 100
        errores = y_pred[mascara & (y_pred != i)]
        confusion = ""
        if len(errores) > 0:
            confusion = f"  (se confunde con {class_names[np.bincount(errores).argmax()]})"
        print(f"  {nombre}: {acc:.1f}%  [{mascara.sum()} imgs]{confusion}")
    print(f"  GLOBAL: {(y_true == y_pred).mean() * 100:.1f}%")


def main():
    revisar_dataset()
    train_ds, val_ds, class_names, class_weight = cargar_datasets()
    modelo, base = construir_modelo(len(class_names))

    checkpoint = ModelCheckpoint(
        MODELO_SALIDA, monitor="val_loss", save_best_only=True, verbose=1)

    def callbacks():
        return [EarlyStopping(monitor="val_loss", patience=8,
                              restore_best_weights=True, verbose=1), checkpoint]

    print("\n--- FASE 1: cabeza clasificadora ---")
    compilar(modelo, 1e-3)
    h1 = modelo.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_CABEZA,
                    class_weight=class_weight, callbacks=callbacks())

    print("\n--- FASE 2: fine-tuning ---")
    base.trainable = True
    for capa in base.layers[:-30]:
        capa.trainable = False
    compilar(modelo, 1e-5)
    h2 = modelo.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
                    class_weight=class_weight, callbacks=callbacks())

    guardar_curvas([h1, h2])
    evaluar_por_clase(val_ds, class_names)
    print("\nModelo:", MODELO_SALIDA)
    print("Clases:", class_names)


if __name__ == "__main__":
    main()