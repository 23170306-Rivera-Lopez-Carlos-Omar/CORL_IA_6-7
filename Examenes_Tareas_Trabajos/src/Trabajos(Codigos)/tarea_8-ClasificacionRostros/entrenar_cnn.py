"""
Tarea 8 - Entrenamiento de una CNN para reconocimiento facial (alumnos + famosos).

Dos arquitecturas a elegir con la variable ARQUITECTURA:
  "propia"    -> CNN desde cero (Conv + BatchNorm + MaxPooling + Dropout)
  "mobilenet" -> Transfer Learning con MobileNetV2 preentrenada en ImageNet

Estructura esperada del dataset (una carpeta por persona, alumnos y famosos):
    dataset/
        Ana/ ...jpg
        Beto/ ...jpg
        Messi/ ...jpg

Genera:  modelo_<ARQUITECTURA>.keras, clases.json, curvas_<ARQUITECTURA>.png

Uso:  python entrenar_cnn.py
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

# ----------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------
ARQUITECTURA = "propia"        # "propia" o "mobilenet"
AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_DATASET = os.path.join(AQUI, "dataset")
TAMANO_IMG = (128, 128)        # en tiempo real se debe usar el MISMO tamano
BATCH = 32
EPOCAS = 40                    # maximo; EarlyStopping se detiene antes si ya no mejora

ARCHIVO_MODELO = os.path.join(AQUI, f"modelo_{ARQUITECTURA}.keras")

# ----------------------------------------------------------------------
# 1. Datos: redimensionado uniforme, etiquetas one-hot,
#    80% entrenamiento / 20% validacion (sin conjunto de prueba).
# ----------------------------------------------------------------------
entrenamiento, validacion = tf.keras.utils.image_dataset_from_directory(
    DIR_DATASET,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=TAMANO_IMG,
    batch_size=BATCH,
    label_mode="categorical",       # one-hot, para categorical_crossentropy
)
clases = entrenamiento.class_names
print(f"Clases ({len(clases)}): {clases}")

with open(os.path.join(AQUI, "clases.json"), "w", encoding="utf-8") as f:
    json.dump(clases, f, ensure_ascii=False, indent=2)

entrenamiento = entrenamiento.cache().shuffle(1000, seed=123).prefetch(tf.data.AUTOTUNE)
validacion = validacion.cache().prefetch(tf.data.AUTOTUNE)

# Balanceo: si una persona tiene menos fotos, pesa mas al calcular la perdida
# peso = total / (n_clases * fotos_de_la_clase)
conteos = [len(os.listdir(os.path.join(DIR_DATASET, c))) for c in clases]
pesos_clase = {i: sum(conteos) / (len(clases) * n) for i, n in enumerate(conteos)}
for c, n, p in zip(clases, conteos, pesos_clase.values()):
    print(f"  {c:<18} {n:>4} fotos -> peso {p:.2f}")

# ----------------------------------------------------------------------
# 2. Aumento de datos (solo actua al entrenar; en tiempo real no hace nada)
# ----------------------------------------------------------------------
aumento = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="aumento_de_datos")


# ----------------------------------------------------------------------
# 3. Arquitecturas
# ----------------------------------------------------------------------
def bloque_conv(x, filtros):
    """(Conv -> BatchNorm -> ReLU) x2 -> MaxPooling."""
    for _ in range(2):
        x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Activation("relu")(x)
    return layers.MaxPooling2D(2)(x)


def construir_propia():
    """CNN desde cero, con la estructura del tutorial (Conv + Pool) mas BatchNorm y Dropout."""
    entrada = layers.Input(shape=TAMANO_IMG + (3,))
    x = aumento(entrada)
    x = layers.Rescaling(1.0 / 255)(x)       # normalizacion a [0, 1]
    x = bloque_conv(x, 32)                   # 128 -> 64
    x = bloque_conv(x, 64)                   #  64 -> 32
    x = bloque_conv(x, 128)                  #  32 -> 16
    x = bloque_conv(x, 256)                  #  16 -> 8
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    # El Dropout va DESPUES de BatchNorm (antes de una BatchNorm puede romper la validacion)
    x = layers.Dropout(0.5)(x)
    salida = layers.Dense(len(clases), activation="softmax")(x)   # una neurona por persona
    return models.Model(entrada, salida, name="cnn_propia")


def construir_mobilenet():
    """Transfer Learning: MobileNetV2 congelada + cabeza nueva para nuestras clases."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=TAMANO_IMG + (3,),
        include_top=False,        # sin las 1000 clases de ImageNet
        weights="imagenet",       # la primera vez descarga los pesos (~9 MB)
    )
    base.trainable = False        # solo se entrena la parte nueva

    entrada = layers.Input(shape=TAMANO_IMG + (3,))
    x = aumento(entrada)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)   # normalizacion a [-1, 1] (la de MobileNetV2)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(len(clases), activation="softmax")(x)
    return models.Model(entrada, salida, name="mobilenetv2_rostros")


model = construir_propia() if ARQUITECTURA == "propia" else construir_mobilenet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",   # clasificacion multiclase con etiquetas one-hot
    metrics=["accuracy"],
)
model.summary()

# ----------------------------------------------------------------------
# 4. Entrenamiento con callbacks
# ----------------------------------------------------------------------
history = model.fit(
    entrenamiento,
    validation_data=validacion,
    epochs=EPOCAS,
    class_weight=pesos_clase,
    callbacks=[
        # Detiene si val_loss no mejora en 8 epocas y regresa a los mejores pesos
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        # Guarda en disco solo el mejor modelo (menor val_loss)
        callbacks.ModelCheckpoint(ARCHIVO_MODELO, monitor="val_loss", save_best_only=True, verbose=1),
        # Si se estanca, reduce la tasa de aprendizaje a la mitad
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ],
)

perdida, precision = model.evaluate(validacion, verbose=0)
print(f"\nMejor modelo -> val_loss: {perdida:.4f} | val_accuracy: {precision:.2%}")
print(f"Guardado en {os.path.basename(ARCHIVO_MODELO)}")

# ----------------------------------------------------------------------
# 5. Matriz de confusion (filas = persona real, columnas = prediccion)
# ----------------------------------------------------------------------
y_real, y_pred = [], []
for imagenes, etiquetas in validacion:
    y_real.extend(np.argmax(etiquetas.numpy(), axis=1))
    y_pred.extend(np.argmax(model.predict(imagenes, verbose=0), axis=1))
matriz = tf.math.confusion_matrix(y_real, y_pred, num_classes=len(clases)).numpy()

print("\nMatriz de confusion (filas = real, columnas = predicho):")
print(" " * 12 + "".join(f"{c[:10]:>11}" for c in clases))
for nombre, fila in zip(clases, matriz):
    print(f"{nombre[:10]:>11} " + "".join(f"{v:>11}" for v in fila))

# ----------------------------------------------------------------------
# 6. Curvas de perdida y accuracy (entrenamiento vs validacion)
#    Si entrenamiento sube y validacion no -> sobreajuste.
# ----------------------------------------------------------------------
h = history.history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(h["accuracy"], label="Entrenamiento")
ax1.plot(h["val_accuracy"], label="Validacion")
ax1.set_title("Accuracy")
ax2.plot(h["loss"], label="Entrenamiento")
ax2.plot(h["val_loss"], label="Validacion")
ax2.set_title("Perdida (categorical_crossentropy)")
for ax in (ax1, ax2):
    ax.set_xlabel("Epoca")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
fig.suptitle(f"Arquitectura: {ARQUITECTURA}", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(AQUI, f"curvas_{ARQUITECTURA}.png"), dpi=120)
print(f"Curvas guardadas en curvas_{ARQUITECTURA}.png")
plt.show()
