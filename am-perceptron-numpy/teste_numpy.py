import numpy as np

# Criar os vetores
x = np.array([8.0, 7.0]) # Entradas (x1, x2)
w = np.array([0.8, 0.3]) # Pesos (w1, w2)
bias = -7.0

# Calculo do produto escalar
z_dot = np.dot(x, w) + bias

# Com operador @
z_operador = (x @ w) + bias

print("Z com np.dot: ", z_dot)
print("Z com operador @: ", z_operador)