
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# CONFIGURACION GENERAL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_LOCAL = os.path.join(BASE_DIR, "dataset")

DATASET_TAREA6 = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "tarea_6 -CrearDatasetPropio",
        "dataset"
    )
)

MODELO_SALIDA = os.path.join(
    BASE_DIR,
    "modelo_rostros.keras"
)

CLASES_SALIDA = os.path.join(
    BASE_DIR,
    "clases.json"
)

GRAFICA_SALIDA = os.path.join(
    BASE_DIR,
    "curvas_entrenamiento.png"
)

IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 50
SEED = 42

# LOCALIZAR DATASET

def resolver_dataset():

    if os.path.isdir(DATASET_LOCAL):
        return DATASET_LOCAL

    if os.path.isdir(DATASET_TAREA6):
        return DATASET_TAREA6

    raise FileNotFoundError(
        "No se encontro la carpeta dataset.\n"
        "Verifica la ubicacion del dataset."
    )


# CARGAR DATASET Y DIVIDIRLO

def cargar_datasets(ruta_dataset):

    print("\nCargando imagenes del dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        ruta_dataset,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="training"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        ruta_dataset,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="validation"
    )

    class_names = train_ds.class_names

    if len(class_names) < 2:
        raise ValueError(
            "Se necesitan al menos dos personas diferentes."
        )

    # Guardar nombres de las clases
    with open(
        CLASES_SALIDA,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            class_names,
            archivo,
            indent=2,
            ensure_ascii=False
        )

    # CALCULAR PESOS DE CLASE

    conteos = np.zeros(
        len(class_names),
        dtype=np.float32
    )

    for _, etiquetas in train_ds.unbatch():

        indice = int(
            np.argmax(etiquetas.numpy())
        )

        conteos[indice] += 1

    total = float(np.sum(conteos))
    num_clases = len(class_names)

    class_weight = {}

    for i in range(num_clases):

        if conteos[i] > 0:

            peso = total / (
                num_clases * conteos[i]
            )

            class_weight[i] = float(
                min(peso, 4.0)
            )

        else:

            class_weight[i] = 1.0

    print("\n================================")
    print("INFORMACION DEL DATASET")
    print("================================")

    print("Personas:", class_names)

    print(
        "Imagenes por clase:",
        dict(
            zip(
                class_names,
                conteos.astype(int).tolist()
            )
        )
    )

    print("Pesos aplicados:", class_weight)

    # OPTIMIZAR CARGA DE DATOS

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return (
        train_ds,
        val_ds,
        class_names,
        class_weight
    )


# CONSTRUIR ARQUITECTURA CNN

def construir_cnn(num_clases):

    print("\nConstruyendo arquitectura CNN...")

    # Aumento de datos
    aumento = models.Sequential(
        [
            layers.RandomFlip("horizontal"),

            layers.RandomRotation(0.1),

            layers.RandomZoom(0.12)
        ],
        name="data_augmentation"
    )

    entradas = layers.Input(
        shape=(128, 128, 3)
    )

    # Aplicar aumento de datos
    x = aumento(entradas)

    # Normalizar pixeles entre 0 y 1
    x = layers.Rescaling(
        1.0 / 255.0
    )(x)

    # BLOQUES CONVOLUCIONALES

    for filtros in (32, 64, 128):

        x = layers.Conv2D(
            filtros,
            (3, 3),
            padding="same",
            use_bias=False
        )(x)

        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(
            (2, 2)
        )(x)

    # EXTRACCION DE CARACTERISTICAS

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.Dropout(0.35)(x)

    # CAPA DE CLASIFICACION

    salidas = layers.Dense(
        num_clases,
        activation="softmax"
    )(x)

    modelo = models.Model(
        entradas,
        salidas,
        name="cnn_reconocimiento_facial"
    )

    # COMPILACION

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=8e-4
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )

    return modelo


# GENERAR GRAFICAS

def guardar_curvas(history):

    print("\nGenerando graficas...")

    acc = history.history["accuracy"]

    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]

    val_loss = history.history["val_loss"]

    epocas = range(
        1,
        len(acc) + 1
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )
    # GRAFICA ACCURACY

    ax1.plot(
        epocas,
        acc,
        label="Entrenamiento"
    )

    ax1.plot(
        epocas,
        val_acc,
        label="Validacion"
    )

    ax1.set_title(
        "Precision (Accuracy)"
    )

    ax1.set_xlabel("Epocas")

    ax1.set_ylabel("Accuracy")

    ax1.legend()

    ax1.grid(True)

    # GRAFICA LOSS

    ax2.plot(
        epocas,
        loss,
        label="Entrenamiento"
    )

    ax2.plot(
        epocas,
        val_loss,
        label="Validacion"
    )

    ax2.set_title(
        "Perdida (Loss)"
    )

    ax2.set_xlabel("Epocas")

    ax2.set_ylabel("Loss")

    ax2.legend()

    ax2.grid(True)

    plt.tight_layout()

    plt.savefig(
        GRAFICA_SALIDA,
        dpi=150
    )

    plt.close(fig)

    print("Grafica guardada:", GRAFICA_SALIDA)


# ENTRENAMIENTO PRINCIPAL
def main():

    print("====================================")

    print(" SISTEMA DE RECONOCIMIENTO FACIAL")

    print(" RED NEURONAL CONVOLUCIONAL CNN")

    print("====================================")

    # Localizar dataset
    ruta_dataset = resolver_dataset()

    print("\nDataset utilizado:")
    print(ruta_dataset)

    # Cargar datos
    (
        train_ds,
        val_ds,
        class_names,
        class_weight
    ) = cargar_datasets(ruta_dataset)

    # Construir modelo
    modelo = construir_cnn(
        len(class_names)
    )

    # Mostrar arquitectura
    print("\nRESUMEN DE LA CNN")

    modelo.summary()

    # CALLBACKS

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=MODELO_SALIDA,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # ENTRENAR

    print("\nIniciando entrenamiento...")

    history = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks
    )


    # GUARDAR RESULTADOS

    guardar_curvas(history)

    print("\n====================================")

    print("ENTRENAMIENTO FINALIZADO")

    print("====================================")

    print("Modelo:", MODELO_SALIDA)

    print("Clases:", CLASES_SALIDA)

    print("Graficas:", GRAFICA_SALIDA)

    print("\nPersonas reconocibles:")

    for persona in class_names:
        print("-", persona)


if __name__ == "__main__":
    main()
