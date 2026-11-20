import numpy as np

# 1. Definimos las compuertas lógicas 
# Entradas (X1, X2) y salida esperada (Y)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([0, 0, 0, 1]) # El AND solo da 1 si ambos son 1

# 2. Inicializamos los pesos y el sesgo (bias) en ceros o valores aleatorios pequeños
np.random.seed(42)
pesos = np.random.randn(2)
bias = np.random.randn()
learning_rate = 0.1
epochs = 10

print("--- ENTRENAMIENTO DEL PERCEPTRÓN (Compuerta AND) ---")

# 3. Función de activación (Escalón / Step function)
def activar(x):
    return 1 if x >= 0 else 0

# 4. Entrenamiento 
for epoch in range(epochs):
    error_total = 0
    for i in range(len(X)):
        # Calcular la suma ponderada: z = (w1*x1 + w2*x2) + bias
        z = np.dot(X[i], pesos) + bias
        prediccion = activar(z)
        
        # Calcular el error (Salida Real - Predicción)
        error = Y[i] - prediccion
        error_total += abs(error)
        
        # Actualizar pesos y bias si hubo error
        pesos += learning_rate * error * X[i]
        bias += learning_rate * error
        
    print(f"Época {epoch+1} - Pesos: {pesos}, Bias: {bias:.2f}")
    if error_total == 0:
        print("¡El perceptrón aprendió la regla perfectamente!\n")
        break

# 5. Probando el modelo entrenado
print("---|||| PRUEBAS FINALES ||||---")
for x in X:
    resultado = activar(np.dot(x, pesos) + bias)
    print(f"Entrada: {x} -> Predicción de la Neurona: {resultado}")