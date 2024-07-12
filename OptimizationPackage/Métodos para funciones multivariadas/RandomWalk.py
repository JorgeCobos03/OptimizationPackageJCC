"""
Algoritmo de Random Walk
=========================

Este módulo implementa el algoritmo de Random Walk para la optimización. El algoritmo puede ser utilizado para encontrar 
el mínimo de una función en un espacio multidimensional.

Funciones:
----------
- random_walk(f_name_or_index, x0, max_iter=1000, epsilon=1e-6, mu=0, sigma=1)

Ejemplo:
--------
import numpy as np
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

# Uso de ejemplo con la función de Beale
x0 = [0, 0]
mejor_solucion, valor_funcion = random_walk(4, x0)  # Usando el índice de la función
print(f"Mejor solución: {mejor_solucion}, Valor de la función: {valor_funcion}")

mejor_solucion, valor_funcion = random_walk('ackley', x0)  # Usando el nombre de la función
print(f"Mejor solución: {mejor_solucion}, Valor de la función: {valor_funcion}")
"""

import numpy as np

def random_walk(f_name_or_index, x0, max_iter=1000, epsilon=1e-6, mu=0, sigma=1):
    """
    Implementa el algoritmo de Random Walk para minimizar la función objetivo.

    Parámetros
    ----------
    f_name_or_index : str or int
        El nombre de la función objetivo a minimizar o su índice en benchmark_functions.
    x0 : list or np.array
        La solución inicial.
    max_iter : int, opcional
        El número máximo de iteraciones (por defecto es 1000).
    epsilon : float, opcional
        Criterio de terminación basado en la tolerancia (por defecto es 1e-6).
    mu : float, opcional
        Media de la distribución normal para la generación de pasos aleatorios (por defecto es 0).
    sigma : float, opcional
        Desviación estándar de la distribución normal para la generación de pasos aleatorios (por defecto es 1).

    Retorna
    -------
    list
        La mejor solución encontrada.
    float
        El valor de la función objetivo en la mejor solución.
    """
    # Seleccionar la función objetivo
    if isinstance(f_name_or_index, str):
        f = getattr(bf, f_name_or_index)
    else:
        f = benchmark_functions[f_name_or_index]

    x_best = np.array(x0)
    f_best = f(x_best)
    
    for _ in range(max_iter):
        # Generación del paso aleatorio
        x_next = x_best + np.random.normal(mu, sigma, len(x0))
        
        # Evaluar la nueva solución
        f_next = f(x_next)
        
        # Actualizar la mejor solución encontrada
        if f_next < f_best:
            x_best = x_next
            f_best = f_next
            
        # Criterio de terminación
        if abs(f_next - f_best) < epsilon:
            break
    
    return x_best, f_best

# Ejemplo de uso del Random Walk con funciones de benchmark
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

    # Uso de ejemplo con la función de Beale
    x0 = [0, 0]
    mejor_solucion, valor_funcion = random_walk(4, x0)  # Usando el índice de la función
    print(f"Mejor solución: {mejor_solucion}, Valor de la función: {valor_funcion}")

    mejor_solucion, valor_funcion = random_walk('ackley', x0)  # Usando el nombre de la función
    print(f"Mejor solución: {mejor_solucion}, Valor de la función: {valor_funcion}")
