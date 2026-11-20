import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Configuramos el estilo de las gráficas 
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. GENERACIÓN DE DATOS 
print("Cargando dataset de California Housing...")
california = fetch_california_housing(as_frame=True)

# Tomamos el ingreso medio (MedInc) vs el valor de la casa (Target)
# Se toma un subconjunto de 50 para visualizarlo mejor
X = california.data['MedInc'].values[:50] 
Y = california.target.values[:50] 

print(f"Datos cargados con éxito: {len(X)} casas analizadas.")

# 2. INICIALIZACIÓN DE PARÁMETROS
w = 0.0
b = 0.0

# Hiperparámetros
learning_rate = 0.01 
epochs = 1500         
n = len(X)

# Historial para graficar la evolución del costo
historial_costos = []

# 3. ENTRENAMIENTO 
for epoch in range(epochs):
    # Predicción actual
    Y_pred = w * X + b
    
    # Cálculo del Costo (MSE)
    cost = (1 / (2 * n)) * np.sum((Y_pred - Y) ** 2)
    historial_costos.append(cost)
    
    # Derivadas parciales (Gradientes)
    dw = (2 / n) * np.sum((Y_pred - Y) * X)
    db = (2 / n) * np.sum(Y_pred - Y)
    
    # Actualización de pesos
    w = w - learning_rate * dw
    b = b - learning_rate * db

print("¡Entrenamiento completado!")
print(f"Ecuación final aprendida: Valor de Casa = {w:.4f} * (Ingreso Medio) + {b:.4f}")

# 4. VISUALIZACIÓN DE RESULTADOS 
plt.figure(figsize=(10, 6))

# Dibujar los puntos reales
plt.scatter(X, Y, color='royalblue', s=70, label='Datos Reales (Casas)', zorder=5)

# Dibujar la línea de tendencia aprendida por el modelo
X_linea = np.linspace(min(X) - 0.5, max(X) + 0.5, 100)
Y_linea = w * X_linea + b
plt.plot(X_linea, Y_linea, color='crimson', linewidth=3, label=f'Modelo Lineal: Y = {w:.2f}X + {b:.2f}')

# Etiquetas y formato estético
plt.title('Regresión Lineal: Ingreso Medio vs Valor de Casa (California)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Ingreso Medio (MedInc)', fontsize=11)
plt.ylabel('Valor de la Casa (100k$)', fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 5. VISUALIZACIÓN DEL COSTO
plt.figure(figsize=(8, 4))
plt.plot(range(epochs), historial_costos, color='darkorange', linewidth=2)
plt.title('Evolución de la Función de Costo (Error Cuadrático Medio)', fontsize=11, fontweight='bold')
plt.xlabel('Épocas / Iteraciones', fontsize=10)
plt.ylabel('Costo (MSE)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()