# ==============================================================================
# TALLER 11: PERCEPTRÓN DESDE CERO - COMPUERTA OR
# ==============================================================================

import numpy as np

# 1. Definir la Función de Activación (Escalón)
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

# 2. Definir la Estructura de la Neurona
def perceptron(X, W, b):
    # Combinación lineal (Producto punto + sesgo)
    Z = np.dot(X, W) + b
    salida = funcion_escalon(Z)
    return salida

# 3. Configuración de pesos y sesgo para resolver la Compuerta OR
# Pesos y sesgo ajustados manualmente para el comportamiento OR:
pesos = np.array([0.8, 0.8])  # Vector W
sesgo = -0.4                  # Constante b (Umbral de disparo)

# 4. Pruebas de todas las combinaciones posibles de la compuerta OR
casos_prueba = [
    np.array([0, 0]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1])
]

print("--- EVALUANDO COMPUERTA OR ---")
for entrada in casos_prueba:
    resultado = perceptron(entrada, pesos, sesgo)
    print(f"Entrada: {entrada} -> El Perceptrón disparó el valor: {resultado}")
