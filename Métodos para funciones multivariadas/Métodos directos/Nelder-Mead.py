"""
Algoritmo Nelder-Mead Simplex
==============================

Este módulo implementa el algoritmo de optimización Nelder-Mead Simplex. El algoritmo puede ser utilizado para encontrar 
el mínimo de una función en un espacio multidimensional.

Funciones:
----------
- nelder_mead(func, initial_simplex, gamma=2, beta=0.5, sigma=0.5, tol=1e-5, max_iter=1000)

Ejemplo:
--------
import numpy as np
import benchmark_functions as bf

# Uso de ejemplo con la función de Rastrigin
initial_simplex = np.array([[3, 3], [4, 3], [3, 4]])
result = nelder_mead(bf.rastrigin, initial_simplex)
print(f"Punto óptimo: {result}")

"""
import numpy as np

def nelder_mead(func, initial_simplex, gamma=2, beta=0.5, sigma=0.5, tol=1e-5, max_iter=1000):
    """
    Realiza la optimización Nelder-Mead Simplex.

    Parámetros
    ----------
    func : callable
        La función objetivo a minimizar.
    initial_simplex : np.ndarray
        Simplejo inicial (array de puntos).
    gamma : float, opcional
        Parámetro de expansión (por defecto es 2).
    beta : float, opcional
        Parámetro de contracción (por defecto es 0.5).
    sigma : float, opcional
        Parámetro de reducción (por defecto es 0.5).
    tol : float, opcional
        Tolerancia para la terminación (por defecto es 1e-5).
    max_iter : int, opcional
        Número máximo de iteraciones (por defecto es 1000).

    Retorna
    -------
    np.ndarray
        La posición estimada del mínimo.
    """
    simplex = initial_simplex.copy()
    num_points = len(simplex)
    
    for iteration in range(max_iter):
        # Ordenar los puntos del simplex por sus valores de función
        simplex = sorted(simplex, key=lambda x: func(x))
        x_best = simplex[0]
        x_worst = simplex[-1]
        x_second_worst = simplex[-2]

        # Calcular el centroide de los mejores puntos
        x_centroid = np.mean(simplex[:-1], axis=0)

        # Reflexión
        x_reflected = x_centroid + gamma * (x_centroid - x_worst)
        if func(x_best) <= func(x_reflected) < func(x_second_worst):
            simplex[-1] = x_reflected
        else:
            if func(x_reflected) < func(x_best):
                # Expansión
                x_expanded = x_centroid + gamma * (x_reflected - x_centroid)
                if func(x_expanded) < func(x_reflected):
                    simplex[-1] = x_expanded
                else:
                    simplex[-1] = x_reflected
            else:
                # Contracción
                x_contracted = x_centroid + beta * (x_worst - x_centroid)
                if func(x_contracted) < func(x_worst):
                    simplex[-1] = x_contracted
                else:
                    # Reducción
                    simplex = [x_best + sigma * (x - x_best) for x in simplex[1:]]
                    simplex.insert(0, x_best)

        # Verificar la convergencia
        if np.std([func(x) for x in simplex]) < tol:
            break

    return simplex[0]

# Ejemplo de uso del Nelder-Mead Simplex con funciones de benchmark
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
    initial_simplex = np.array([[3, 3], [4, 3], [3, 4]])
    result = nelder_mead(benchmark_functions[0], initial_simplex)
    print(f"Punto óptimo: {result}")
