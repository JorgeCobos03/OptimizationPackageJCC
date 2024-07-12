# Métodos para Funciones Multivariadas

Este proyecto implementa varios métodos numéricos para encontrar el mínimo de funciones multivariadas en Python. Se incluyen tanto métodos directos como métodos basados en gradientes. A continuación, se describen los métodos y se proporciona un ejemplo de implementación utilizando el algoritmo Hooke-Jeeves.

## Métodos Directos

* Caminata Aleatoria

El método de caminata aleatoria realiza movimientos aleatorios en el espacio de búsqueda. Es útil para explorar soluciones en problemas donde la función objetivo no es diferenciable o no tiene una estructura regular.

* Método de Nelder y Mead (Simplex)

El método de Nelder y Mead, también conocido como método del simplex, utiliza un poliedro en el espacio de búsqueda para aproximar la solución óptima. Es robusto y no requiere derivadas de la función objetivo.

* Método de Hooke-Jeeves

El método de Hooke-Jeeves es una técnica de búsqueda directa que explora el espacio de búsqueda mediante pasos incrementales y reducciones adaptativas del tamaño del paso. Es efectivo para encontrar mínimos locales en funciones continuas.

## Métodos de Gradiente

* Método de Cauchy

El método de Cauchy utiliza una combinación de descensos por gradiente y pasos de búsqueda lineal para encontrar el mínimo local de una función. Es eficiente pero puede requerir ajustes en el tamaño de paso.

* Método de Fletcher-Reeves

El método de Fletcher-Reeves es un algoritmo de descenso por gradiente conjugado que utiliza direcciones conjugadas para mejorar la convergencia hacia el mínimo local de una función.

* Método de Newton

El método de Newton es un algoritmo avanzado que utiliza la matriz Hessiana de la función objetivo para calcular la dirección y el tamaño del paso óptimos. Es eficiente pero puede ser sensible a la precisión numérica y requerir evaluaciones exactas de la Hessiana.

## Ejemplo de Implementación: Algoritmo Hooke-Jeeves

```python
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

```
## Instrucciones de Uso
1. Instalar Dependencias:
Asegúrese de tener las bibliotecas necesarias instaladas. Puede instalar numpy con el siguiente comando en la terminal:

## Copiar código
```bash
pip install numpy
```
2. Definir Funciones de Benchmark:
Cree las funciones que desea minimizar y guárdelas en un archivo benchmark_functions.py. Asegúrese de que estas funciones estén disponibles para importar en su script principal.

3. Inicializar la Búsqueda:
Cree instancias de los métodos de optimización con las funciones de benchmark y los parámetros de inicialización adecuados según la técnica seleccionada.

4. Realizar la Búsqueda:
Llame al método principal (hooke_jeeves, por ejemplo) con los valores de precisión y otros parámetros necesarios para comenzar la optimización.

5. Visualizar Resultados:
Los resultados se mostrarán en formato de salida especificado en su script principal. Asegúrese de verificar que los valores óptimos encontrados sean los esperados y correspondan a los mínimos locales de las funciones definidas.

