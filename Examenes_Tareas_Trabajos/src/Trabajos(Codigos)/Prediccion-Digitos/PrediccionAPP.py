import os
import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model

MODEL_PATH = "modelo_mnist.keras"

# 1. Modelo: se entrena automáticamente la primera vez y se guarda
def obtener_modelo():
    if os.path.exists(MODEL_PATH):
        print("Cargando modelo guardado...")
        return load_model(MODEL_PATH)

    print("Entrenando modelo CNN (solo la primera vez, toma unos segundos)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
    x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

    # Red Convolucional (CNN): Ideal para reconocer formas y trazos a mano
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
                  
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")
    model.save(MODEL_PATH)
    return model

# 2. Preprocesado: recorta y centra matemáticamente el dibujo como MNIST
def preparar_imagen(img_pil):
    arr = np.array(img_pil)
    ys, xs = np.where(arr > 30)
    if len(xs) == 0:
        return None  # Lienzo vacío

    # Recortar al área exacta del dígito dibujado
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape

    # Escalar para que el lado mayor mida 20 px manteniendo proporciones
    escala = 20.0 / max(h, w)
    nuevo_w, nuevo_h = max(1, int(round(w * escala))), max(1, int(round(h * escala)))
    digito = Image.fromarray(arr).resize((nuevo_w, nuevo_h), Image.LANCZOS)

    # Pegarlo centrado en un lienzo negro de 28x28
    lienzo = Image.new("L", (28, 28), 0)
    lienzo.paste(digito, ((28 - nuevo_w) // 2, (28 - nuevo_h) // 2))

    # Centrar por centro de masa para máxima precisión de la IA
    a = np.array(lienzo).astype("float32")
    total = a.sum()
    if total > 0:
        cy, cx = (np.indices(a.shape) * a).sum(axis=(1, 2)) / total
        dx, dy = int(round(14 - cx)), int(round(14 - cy))
        a = np.roll(a, shift=(dy, dx), axis=(0, 1))

    return (a / 255.0).reshape(1, 28, 28, 1)

# 3. Interfaz Gráfica (Tkinter)
class App:
    TAM = 280      # Tamaño de la pizarra en pantalla
    GROSOR = 18    # Grosor del pincel

    def __init__(self, root, model):
        self.model = model
        self.root = root
        root.title("Clasificador de Dígitos (CNN + Interfaz)")
        root.geometry("320x400")
        root.resizable(False, False)

        # Lienzo para dibujar
        self.canvas = tk.Canvas(root, width=self.TAM, height=self.TAM,
                                bg="black", cursor="cross")
        self.canvas.pack(padx=10, pady=10)

        # Imagen paralela en memoria que lee el modelo
        self.imagen = Image.new("L", (self.TAM, self.TAM), 0)
        self.draw = ImageDraw.Draw(self.imagen)

        self.canvas.bind("<B1-Motion>", self.pintar)

        # Botones de control
        frame_botones = tk.Frame(root)
        frame_botones.pack(pady=5)

        tk.Button(frame_botones, text="Predecir", command=self.predecir, width=12, bg="#d1e7dd").pack(side=tk.LEFT, padx=5)
        tk.Button(frame_botones, text="Borrar", command=self.borrar, width=12, bg="#f8d7da").pack(side=tk.RIGHT, padx=5)

        # Textos de resultados
        self.resultado = tk.Label(root, text="Dibuja un número y presiona Predecir", font=("Arial", 11, "bold"))
        self.resultado.pack(pady=5)

        self.detalle = tk.Label(root, text="", font=("Arial", 9), fg="gray")
        self.detalle.pack()

    def pintar(self, event):
        r = self.GROSOR // 2
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def borrar(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.TAM, self.TAM], fill=0)
        self.resultado.config(text="Dibuja un número y presiona Predecir")
        self.detalle.config(text="")

    def predecir(self):
        entrada = preparar_imagen(self.imagen)
        if entrada is None:
            self.resultado.config(text="El lienzo está vacío")
            return
            
        probs = self.model.predict(entrada, verbose=0)[0]
        orden = np.argsort(probs)[::-1]
        mejor = orden[0]
        
        self.resultado.config(
            text=f"Es un: {mejor} ({probs[mejor] * 100:.1f}% de confianza)")
        self.detalle.config(
            text=f"2do opción: {orden[1]} ({probs[orden[1]] * 100:.1f}%) | 3ro: {orden[2]} ({probs[orden[2]] * 100:.1f}%)")

if __name__ == "__main__":
    modelo = obtener_modelo()
    ventana = tk.Tk()
    App(ventana, modelo)
    ventana.mainloop()