"""
Método de Newton
==============================

Este módulo implementa el método de Newton para la optimización. El algoritmo puede ser utilizado para encontrar 
el mínimo de una función en un espacio multidimensional.

Funciones:
----------
- himmelblau(x)
- gradiente(f, x, deltaX=1e-5)
- hessian_matrix(f, x, deltaX=1e-5)
- newton_method(f, grad_f, x0, tol1=1e-5, tol2=1e-5, tol3=1e-5, max_iter=1000)

Ejemplo:
--------
import benchmark_functions as bf
"""

import numpy as np
import math

# Cálculo numérico del gradiente
def gradiente(f, x, deltaX=1e-5):
    """
    Calcula el gradiente numérico de una función en un punto dado.
    
    :param f: Función de la cual se quiere calcular el gradiente.
    :param x: Lista o tupla con las coordenadas en las que se calcula el gradiente.
    :param deltaX: Incremento pequeño para aproximar la derivada.
    :return: Gradiente numérico de la función en el punto dado.
    """
    grad = []
    for i in range(len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] += deltaX
        xn[i] -= deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return np.array(grad)

# Cálculo numérico de la matriz Hessiana
def hessian_matrix(f, x, deltaX=1e-5):
    """
    Calcula la matriz Hessiana numérica de una función en un punto dado.
    
    :param f: Función de la cual se quiere calcular la matriz Hessiana.
    :param x: Lista o tupla con las coordenadas en las que se calcula la matriz Hessiana.
    :param deltaX: Incremento pequeño para aproximar las segundas derivadas.
    :return: Matriz Hessiana numérica de la función en el punto dado.
    """
    fx = f(x)
    N = len(x)
    H = []
    for i in range(N):
        hi = []
        for j in range(N):
            if i == j:
                xp = x.copy()
                xn = x.copy()
                xp[i] += deltaX
                xn[i] -= deltaX
                hi.append((f(xp) - 2*fx + f(xn)) / (deltaX**2))
            else:
                xpp = x.copy()
                xpn = x.copy()
                xnp = x.copy()
                xnn = x.copy()
                xpp[i] += deltaX
                xpp[j] += deltaX
                xpn[i] += deltaX
                xpn[j] -= deltaX
                xnp[i] -= deltaX
                xnp[j] += deltaX
                xnn[i] -= deltaX
                xnn[j] -= deltaX
                hi.append((f(xpp) - f(xpn) - f(xnp) + f(xnn)) / (4 * deltaX**2))
        H.append(hi)
    return np.array(H)

# Método de Newton
def newton_method(f, grad_f, x0, tol1=1e-5, tol2=1e-5, tol3=1e-5, max_iter=1000):
    """
    Implementa el método de Newton para minimizar una función.
    
    :param f: Función a minimizar.
    :param grad_f: Gradiente de la función.
    :param x0: Punto inicial.
    :param tol1: Tolerancia para el gradiente.
    :param tol2: Tolerancia para el cambio en x.
    :param tol3: Tolerancia para el valor del gradiente en el nuevo punto.
    :param max_iter: Número máximo de iteraciones.
    :return: Punto que minimiza la función.
    """
    x = np.array(x0)
    
    for k in range(max_iter):
        grad = grad_f(x)
        H = hessian_matrix(f, x)
        
        # Calcular la dirección de Newton
        H_inv = np.linalg.inv(H)
        p = -np.dot(H_inv, grad)
        
        # Búsqueda unidireccional para encontrar α
        def f_alpha(alpha):
            return f(x + alpha * p)
        
        # Método de búsqueda unidireccional manual (Golden Section)
        def golden_section_search(f, a, b, tol=1e-5):
            phi = (1 + math.sqrt(5)) / 2
            resphi = 2 - phi
            x1 = a + resphi * (b - a)
            x2 = b - resphi * (b - a)
            f1 = f(x1)
            f2 = f(x2)
            while abs(b - a) > tol:
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = a + resphi * (b - a)
                    f1 = f(x1)
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = b - resphi * (b - a)
                    f2 = f(x2)
            return (a + b) / 2

        alpha = golden_section_search(f_alpha, 0, 1)
        
        # Actualizar x
        x_new = x + alpha * p
        
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            norm_x = 1  # Evitar división por cero
        
        if np.linalg.norm(x_new - x) / norm_x < tol2 or np.linalg.norm(grad_f(x_new)) <= tol3:
            x = x_new
            break
        
        x = x_new
        
    return x

# Ejemplo de uso:
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

    # Uso de ejemplo con la función de Himmelblau
    x0 = [0.0, 0.0]
    minimo = newton_method(benchmark_functions[10], lambda x: gradiente(benchmark_functions[10], x), x0)
    print(f"Resultado Método de Newton: {minimo}")
