"""
Algoritmo de Gradiente Conjugado
==============================

Este módulo implementa el algoritmo de optimización de gradiente conjugado. El algoritmo puede ser utilizado para encontrar 
el mínimo de una función en un espacio multidimensional.

Funciones:
----------
- himmelblau(x)
- gradiente(f, x, deltaX=0.001)
- busqueda_unidireccional(f_lambda, a=0, b=1, tol=1e-5)
- conjugate_gradient_method(f, x0, tol1=1e-5, tol2=1e-5, tol3=1e-5, max_iter=1000)

Ejemplo:
--------
import numpy as np
import benchmark_functions as bf
"""

import math

# Gradiente numérico
def gradiente(f, x, deltaX=0.001):
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
    return grad

# Búsqueda unidireccional manual
def busqueda_unidireccional(f_lambda, a=0, b=1, tol=1e-5):
    """
    Realiza una búsqueda unidireccional para minimizar una función en un intervalo dado.
    
    :param f_lambda: Función unidimensional a minimizar.
    :param a: Límite inferior del intervalo.
    :param b: Límite superior del intervalo.
    :param tol: Tolerancia para la convergencia.
    :return: Punto en el intervalo [a, b] que minimiza la función.
    """
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    
    while Lw > tol:
        w2 = aw + PHI * Lw
        w1 = bw - PHI * Lw
        aw, bw = regla_eliminacion(w1, w2, f_lambda(w_to_x(w1, a, b)), f_lambda(w_to_x(w2, a, b)), aw, bw)
        Lw = bw - aw
        
    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

def regla_eliminacion(x1, x2, fx1, fx2, a, b):
    """
    Regla de eliminación para la búsqueda unidireccional.
    
    :param x1: Primer punto en el intervalo.
    :param x2: Segundo punto en el intervalo.
    :param fx1: Valor de la función en x1.
    :param fx2: Valor de la función en x2.
    :param a: Límite inferior del intervalo.
    :param b: Límite superior del intervalo.
    :return: Nuevos límites del intervalo después de la eliminación.
    """
    if fx1 > fx2:
        return x1, b
    if fx1 < fx2:
        return a, x2
    return x1, x2

def w_to_x(w, a, b):
    """
    Convierte un valor w en el intervalo [0, 1] a un valor en el intervalo [a, b].
    
    :param w: Valor en el intervalo [0, 1].
    :param a: Límite inferior del intervalo original.
    :param b: Límite superior del intervalo original.
    :return: Valor correspondiente en el intervalo [a, b].
    """
    return w * (b - a) + a

# Método del gradiente conjugado
def conjugate_gradient_method(f, x0, tol1=1e-5, tol2=1e-5, tol3=1e-5, max_iter=1000):
    """
    Implementa el método del gradiente conjugado para minimizar una función.
    
    :param f: Función a minimizar.
    :param x0: Punto inicial.
    :param tol1: Tolerancia para el cambio en x.
    :param tol2: Tolerancia relativa para el cambio en x.
    :param tol3: Tolerancia para el gradiente.
    :param max_iter: Número máximo de iteraciones.
    :return: Punto que minimiza la función y el valor de la función en ese punto.
    """
    x = x0[:]
    grad = gradiente(f, x)
    s = [-g for g in grad]
    
    for k in range(max_iter):
        # Búsqueda de línea para encontrar λ
        def f_lambda(lmbda):
            x_new = [x[i] + lmbda * s[i] for i in range(len(x))]
            return f(x_new)
        
        lmbda = busqueda_unidireccional(f_lambda)
        
        # Actualizar x
        x_new = [x[i] + lmbda * s[i] for i in range(len(x))]
        
        if all(abs(x_new[i] - x[i]) / (abs(x[i]) if abs(x[i]) > tol1 else 1) < tol2 for i in range(len(x))) \
                or math.sqrt(sum(g**2 for g in gradiente(f, x_new))) <= tol3:
            break
        
        grad_new = gradiente(f, x_new)
        beta = sum(grad_new[i]**2 for i in range(len(x))) / sum(grad[i]**2 for i in range(len(x)))
        
        s = [-grad_new[i] + beta * s[i] for i in range(len(x))]
        x, grad = x_new, grad_new
        
    return x, f(x)

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

    x0 = [0.0, 0.0]
    resultado, valor = conjugate_gradient_method(benchmark_functions[10], x0)
    print(f"Resultado Gradiente Conjugado: {resultado}, Valor de la función: {valor}")
