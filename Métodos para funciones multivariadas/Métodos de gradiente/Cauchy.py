"""
Algoritmo de Cauchy
==============================

Este módulo implementa el algoritmo de optimización de Cauchy. El algoritmo puede ser utilizado para encontrar 
el mínimo de una función en un espacio multidimensional.

Funciones:
----------
- regla_eliminacion(x1, x2, fx1, fx2, a, b)
- w_to_x(w, a, b)
- busquedaDorada(funcion, epsilon, a, b)
- gradiente(f, x, deltaX=0.001)
- cauchy(funcion, x0, epsilon1, epsilon2, M, optimizador_univariable)

Ejemplo:
--------
import numpy as np
import benchmark_functions as bf

# Uso de ejemplo con la función de Himmelblau
x0 = [0.0, 0.0]
resultado, valor = cauchy(bf.himmelblau, x0, 0.001, 0.001, 100, busquedaDorada)
print(f"Resultado Cauchy: {resultado}, Valor de la función: {valor}")
"""

import math
import numpy as np

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

def busquedaDorada(funcion, epsilon, a, b):
    """
    Realiza una búsqueda dorada para minimizar una función en un intervalo dado.
    
    :param funcion: Función unidimensional a minimizar.
    :param epsilon: Tolerancia para la convergencia.
    :param a: Límite inferior del intervalo.
    :param b: Límite superior del intervalo.
    :return: Punto en el intervalo [a, b] que minimiza la función.
    """
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1
    
    while Lw > epsilon:
        w2 = aw + PHI * Lw
        w1 = bw - PHI * Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), funcion(w_to_x(w2, a, b)), aw, bw)
        k += 1
        Lw = bw - aw
        
    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

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

def cauchy(funcion, x0, epsilon1, epsilon2, M, optimizador_univariable):
    """
    Implementa el método de Cauchy para minimizar una función.
    
    :param funcion: Función a minimizar.
    :param x0: Punto inicial.
    :param epsilon1: Tolerancia para el gradiente.
    :param epsilon2: Tolerancia para el cambio en x.
    :param M: Número máximo de iteraciones.
    :param optimizador_univariable: Método de búsqueda unidireccional.
    :return: Punto que minimiza la función y el valor de la función en ese punto.
    """
    terminar = False
    xk = x0
    k = 0
    
    while not terminar:
        grad = gradiente(funcion, xk)
        
        if math.sqrt(sum(g**2 for g in grad)) < epsilon1 or k >= M:
            terminar = True
        else:
            def alpha_function(alpha):
                return funcion([xk[i] - alpha * grad[i] for i in range(len(xk))])
            
            alpha = optimizador_univariable(alpha_function, epsilon=epsilon2, a=0.0, b=1.0)
            x_k1 = [xk[i] - alpha * grad[i] for i in range(len(xk))]
            
            if math.sqrt(sum((x_k1[i] - xk[i])**2 for i in range(len(xk)))) / (math.sqrt(sum(xk[i]**2 for i in range(len(xk)))) + 1e-5) <= epsilon2:
                terminar = True
            else:
                k += 1
                xk = x_k1
    
    return xk, funcion(xk)

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
    resultado, valor = cauchy(benchmark_functions[10], x0, 0.001, 0.001, 100, busquedaDorada)
    print(f"Resultado Cauchy: {resultado}, Valor de la función: {valor}")
