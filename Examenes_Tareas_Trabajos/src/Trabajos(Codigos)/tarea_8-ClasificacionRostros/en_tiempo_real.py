
import json
import os
import cv2
import numpy as np
import tensorflow as tf


# CONFIGURACION

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELO_RUTA = os.path.join(
    BASE_DIR,
    "modelo_rostros.keras"
)

CLASES_RUTA = os.path.join(
    BASE_DIR,
    "clases.json"
)

IMG_SIZE = (128, 128)

CAMARA = 0

UMBRAL_CONFIANZA = 0.55


# CARGAR DETECTOR FACIAL HAAR CASCADE

def cargar_detector():

    # Comprobar que OpenCV tenga el detector
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "OpenCV no tiene disponible CascadeClassifier. "
            "Reinstala opencv-contrib-python."
        )

    # Localizar archivo Haar Cascade
    ruta_xml = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )

    if not os.path.isfile(ruta_xml):
        raise FileNotFoundError(
            f"No se encontro el detector facial: {ruta_xml}"
        )

    # Crear detector
    detector = cv2.CascadeClassifier(ruta_xml)

    if detector.empty():
        raise RuntimeError(
            "No se pudo inicializar Haar Cascade."
        )

    print("Detector facial Haar Cascade cargado correctamente.")

    return detector

# PREDECIR IDENTIDAD DEL ROSTRO

def predecir_rostro(
    modelo,
    clases,
    frame,
    x,
    y,
    w,
    h
):

    alto, ancho = frame.shape[:2]

    # Limites de la region facial
    x1 = max(0, x)

    y1 = max(0, y)

    x2 = min(ancho, x + w)

    y2 = min(alto, y + h)

    # Extraer rostro
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:

        return (
            "No reconocida",
            0.0,
            "Sin segunda clase",
            0.0
        )

    # BGR a RGB
    rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    # Redimensionar
    redimensionada = cv2.resize(
        rgb,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Convertir a float32

    lote = np.expand_dims(
        redimensionada.astype("float32"),
        axis=0
    )

    # Prediccion
    predicciones = modelo.predict(
        lote,
        verbose=0
    )[0]

    # Ordenar probabilidades
    orden = np.argsort(
        predicciones
    )[::-1]

    idx1 = int(orden[0])

    idx2 = (
        int(orden[1])
        if len(orden) > 1
        else idx1
    )

    nombre = clases[idx1]

    confianza = float(
        predicciones[idx1]
    )

    segunda = clases[idx2]

    confianza2 = float(
        predicciones[idx2]
    )

    return (
        nombre,
        confianza,
        segunda,
        confianza2
    )


# DIBUJAR RESULTADOS EN PANTALLA

def dibujar_etiqueta(
    frame,
    x,
    y,
    w,
    h,
    texto1,
    texto2,
    color
):

    alto, ancho = frame.shape[:2]

    # Bounding box
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    # Posicion de etiqueta
    pos_x = max(5, x)

    pos_y = y - 42

    if pos_y < 5:

        pos_y = min(
            y + h + 8,
            alto - 48
        )

    # Limitar ancho de etiqueta
    box_x2 = min(
        ancho - 5,
        pos_x + 345
    )

    # Fondo
    cv2.rectangle(
        frame,
        (pos_x, pos_y),
        (box_x2, pos_y + 43),
        (20, 20, 20),
        -1
    )

    # Primera prediccion
    cv2.putText(
        frame,
        texto1,
        (pos_x + 5, pos_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2
    )

    # Segunda prediccion
    cv2.putText(
        frame,
        texto2,
        (pos_x + 5, pos_y + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1
    )


# FUNCION PRINCIPAL

def main():

    print("======================================")

    print(" RECONOCIMIENTO FACIAL CON CNN")

    print("======================================")

    # VERIFICAR ARCHIVOS
    

    if not os.path.isfile(MODELO_RUTA):

        raise FileNotFoundError(
            "No existe modelo_rostros.keras.\n"
            "Ejecuta primero entrenar_cnn.py"
        )

    if not os.path.isfile(CLASES_RUTA):

        raise FileNotFoundError(
            "No existe clases.json.\n"
            "Ejecuta primero entrenar_cnn.py."
        )

    # CARGAR MODELO

    print("\nCargando modelo CNN...")

    modelo = tf.keras.models.load_model(
        MODELO_RUTA
    )

    # CARGAR CLASES

    with open(
        CLASES_RUTA,
        "r",
        encoding="utf-8"
    ) as archivo:

        clases = json.load(archivo)

    if modelo.output_shape[-1] != len(clases):

        raise ValueError(
            "El numero de clases no coincide con el modelo."
        )

    print("Clases cargadas:", clases)

    # CARGAR DETECTOR
    
    detector = cargar_detector()

    # ABRIR CAMARA
    
    cap = cv2.VideoCapture(CAMARA)

    if not cap.isOpened():

        raise RuntimeError(
            "No se pudo abrir la camara web."
        )

    print("\nCamara iniciada correctamente.")

    print("Presiona Q o ESC para salir.")

    # BUCLE DE VIDEO
    
    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                print(
                    "No se pudo obtener imagen de la camara."
                )

                break

            # Efecto espejo
            frame = cv2.flip(
                frame,
                1
            )

            # Convertir a escala de grises
            gris = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # DETECCION DE ROSTROS
         
            detecciones = detector.detectMultiScale(
                gris,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(60, 60)
            )

            # Mostrar cantidad de rostros
            cv2.putText(
                frame,
                f"Rostros detectados: {len(detecciones)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            # SIN ROSTROS
           
            if len(detecciones) == 0:

                cv2.putText(
                    frame,
                    "No se detecta ningun rostro",
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )

            # PROCESAR ROSTROS
          
            for (x, y, w, h) in detecciones:

                (
                    nombre,
                    confianza,
                    segunda,
                    confianza2
                ) = predecir_rostro(
                    modelo,
                    clases,
                    frame,
                    x,
                    y,
                    w,
                    h
                )

                # EVALUAR CONFIANZA
               
                if confianza >= UMBRAL_CONFIANZA:

                    texto1 = (
                        f"{nombre}: "
                        f"{confianza * 100:.1f}%"
                    )

                    color = (0, 255, 0)

                else:

                    texto1 = (
                        "No reconocida: "
                        f"{confianza * 100:.1f}%"
                    )

                    color = (0, 165, 255)

                texto2 = (
                    f"2da: {segunda} "
                    f"({confianza2 * 100:.1f}%)"
                )

                # Dibujar resultado
                dibujar_etiqueta(
                    frame,
                    x,
                    y,
                    w,
                    h,
                    texto1,
                    texto2,
                    color
                )

            # MOSTRAR CAMARA
         
            cv2.imshow(
                "Reconocimiento Facial CNN - Tarea 8",
                frame
            )

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q") or tecla == 27:

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print("\nCamara cerrada.")


# EJECUCION

if __name__ == "__main__":

    main()
