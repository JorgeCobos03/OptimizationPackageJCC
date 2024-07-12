"""
Algoritmo Hooke-Jeeves
==============================

Este módulo implementa el algoritmo de optimización Hooke-Jeeves. El algoritmo puede ser utilizado para encontrar 
el mínimo de una función en un espacio multidimensional.

Funciones:
----------
- hooke_jeeves(x_initial, delta, alpha, function, max_iterations=1000, tolerance=1e-6)

Ejemplo:
--------
import numpy as np
import benchmark_functions as bf

# Uso de ejemplo con la función de Rastrigin
x_initial = [0, 0]
delta = 0.5
alpha = 0.5
result, value, path = hooke_jeeves(x_initial, delta, alpha, bf.rastrigin)
print(f"Punto óptimo: {result}, Valor de la función: {value}")

"""

import numpy as np

def hooke_jeeves(x_initial, delta, alpha, function, max_iterations=1000, tolerance=1e-6):
    """
    Realiza la optimización Hooke-Jeeves.

    Parámetros
    ----------
    x_initial : list or np.array
        La solución inicial.
    delta : float
        Paso de búsqueda.
    alpha : float
        Factor de reducción del paso.
    function : callable
        La función objetivo a minimizar.
    max_iterations : int, opcional
        Número máximo de iteraciones (por defecto es 1000).
    tolerance : float, opcional
        Tolerancia para la terminación (por defecto es 1e-6).

    Retorna
    -------
    np.ndarray
        La posición estimada del mínimo.
    float
        El valor de la función objetivo en la mejor solución.
    np.ndarray
        La trayectoria de puntos visitados durante la optimización.
    """
    x = np.array(x_initial)
    n = len(x)
    delta_x = np.eye(n) * delta
    f_current = function(x)
    path = [x]

    for _ in range(max_iterations):
        f_best = f_current
        x_best = x.copy()

        for d in range(n):
            x_new = x + delta_x[d]
            f_new = function(x_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new
            else:
                x_new = x - delta_x[d]
                f_new = function(x_new)
                if f_new < f_best:
                    f_best = f_new
                    x_best = x_new

        if f_best >= f_current:
            delta *= alpha
            delta_x = np.eye(n) * delta
        else:
            x = x_best
            f_current = f_best
            path.append(x)

        if np.abs(f_current - f_best) < tolerance:
            break

    return x, f_current, np.array(path)

# Ejemplo de uso del Hooke-Jeeves con funciones de benchmark
if __name__ == "__main__":
    import benchmark_functions as bf

    # Definir las funciones de benchmark
    benchmark_functions = {
        0: bf.rastrigin,
        1: bf.ackley,
        2: bf.sphere,
        3: bf.rosenbrock,
        4: bf.beale,
        5: bf.goldstein_price,
        6: bf.booth,
        7: bf.bukin6,
        8: bf.matyas,
        9: bf.levi13,
        10: bf.himmelblau,
        11: bf.three_hump_camel,
        12: bf.easom,
        13: bf.cross_in_tray,
        14: bf.eggholder,
        15: bf.holder_table,
        16: bf.mccormick,
        17: bf.schaffer2,
        18: bf.schaffer4,
        19: bf.styblinski_tang,
        20: bf.shekel
    }

    # Uso de ejemplo con la función de Rastrigin
    x_initial = [0, 0]
    delta = 0.5
    alpha = 0.5
    result, value, path = hooke_jeeves(x_initial, delta, alpha, benchmark_functions[0])
    print(f"Punto óptimo: {result}, Valor de la función: {value}")
