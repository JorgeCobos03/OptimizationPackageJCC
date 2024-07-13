Métodos para Funciones Multivariadas
====================================

Este proyecto tambien implementa varios métodos numéricos para encontrar el mínimo de funciones multivariadas en Python. Se incluyen tanto métodos directos como métodos basados en gradientes. A continuación, se describen los métodos y se proporciona un ejemplo de implementación utilizando el algoritmo Hooke-Jeeves.


Métodos Directos
----------------

1. Caminata Aleatoria

El método de caminata aleatoria realiza movimientos aleatorios en el espacio de búsqueda. Es útil para explorar soluciones en problemas donde la función objetivo no es diferenciable o no tiene una estructura regular.

Funciones
^^^^^^^^^

.. def:: random_walk(f_name_or_index, x0, max_iter=1000, epsilon=1e-6, mu=0, sigma=1)

  Implementa el algoritmo de Random Walk para minimizar la función objetivo.

  :param f_name_or_index: El nombre de la función objetivo a minimizar o su índice en `benchmark_functions`.
  :type f_name_or_index: str or int
  :param x0: La solución inicial.
  :type x0: list or np.array
  :param max_iter: El número máximo de iteraciones (por defecto es 1000).
  :type max_iter: int, optional
  :param epsilon: Criterio de terminación basado en la tolerancia (por defecto es 1e-6).
  :type epsilon: float, optional
  :param mu: Media de la distribución normal para la generación de pasos aleatorios (por defecto es 0).
  :type mu: float, optional
  :param sigma: Desviación estándar de la distribución normal para la generación de pasos aleatorios (por defecto es 1).
  :type sigma: float, optional

  :returns: La mejor solución encontrada y el valor de la función objetivo en la mejor solución.
  :rtype: tuple

**Ejemplo**

Importar el módulo y definir las funciones de benchmark:

  .. code-block:: python

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

**Código Fuente**

.. code-block:: python

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


2. Método de Nelder y Mead (Simplex)

El método de Nelder y Mead, también conocido como método del simplex, utiliza un poliedro en el espacio de búsqueda para aproximar la solución óptima. Es robusto y no requiere derivadas de la función objetivo.

.. def:: **nelder_mead(func, initial_simplex, gamma=2, beta=0.5, sigma=0.5, tol=1e-5, max_iter=1000)

  Realiza la optimización Nelder-Mead Simplex para minimizar la función objetivo.

  :param func: La función objetivo a minimizar.
  :type func: callable
  :param initial_simplex: Simplejo inicial (array de puntos).
  :type initial_simplex: np.ndarray
  :param gamma: Parámetro de expansión (por defecto es 2).
  :type gamma: float, optional
  :param beta: Parámetro de contracción (por defecto es 0.5).
  :type beta: float, optional
  :param sigma: Parámetro de reducción (por defecto es 0.5).
  :type sigma: float, optional
  :param tol: Tolerancia para la terminación (por defecto es 1e-5).
  :type tol: float, optional
  :param max_iter: Número máximo de iteraciones (por defecto es 1000).
  :type max_iter: int, optional

  :returns: La posición estimada del mínimo encontrado.
  :rtype: np.ndarray

**Código Fuente**

.. code-block:: python

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


3. Método de Hooke-Jeeves

El método de Hooke-Jeeves es una técnica de búsqueda directa que explora el espacio de búsqueda mediante pasos incrementales y reducciones adaptativas del tamaño del paso. Es efectivo para encontrar mínimos locales en funciones continuas.

.. def:: hooke_jeeves(x_initial, delta, alpha, function, max_iterations=1000, tolerance=1e-6)

  Realiza la optimización Hooke-Jeeves para minimizar la función objetivo.

  :param x_initial: La solución inicial.
  :type x_initial: list or np.ndarray
  :param delta: Paso de búsqueda.
  :type delta: float
  :param alpha: Factor de reducción del paso.
  :type alpha: float
  :param function: La función objetivo a minimizar.
  :type function: callable
  :param max_iterations: Número máximo de iteraciones (por defecto es 1000).
  :type max_iterations: int, optional
  :param tolerance: Tolerancia para la terminación (por defecto es 1e-6).
  :type tolerance: float, optional

  :returns: La posición estimada del mínimo encontrado, el valor de la función objetivo en la mejor solución, y la trayectoria de puntos visitados.
  :rtype: np.ndarray, float, np.ndarray

**codigo fuente**

.. code-block:: python
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

Métodos de Gradiente
--------------------

1. Método de Cauchy

El método de Cauchy utiliza una combinación de descensos por gradiente y pasos de búsqueda lineal para encontrar el mínimo local de una función. Es eficiente pero puede requerir ajustes en el tamaño de paso.

.. def:: regla_eliminacion(x1, x2, fx1, fx2, a, b)

  Implementa la regla de eliminación para la búsqueda unidireccional.
  
  :param x1: Primer punto en el intervalo.
  :type x1: float
  :param x2: Segundo punto en el intervalo.
  :type x2: float
  :param fx1: Valor de la función en x1.
  :type fx1: float
  :param fx2: Valor de la función en x2.
  :type fx2: float
  :param a: Límite inferior del intervalo.
  :type a: float
  :param b: Límite superior del intervalo.
  :type b: float
  :returns: Nuevos límites del intervalo después de la eliminación.
  :rtype: tuple

.. def:: w_to_x(w, a, b)

  Convierte un valor w en el intervalo [0, 1] a un valor en el intervalo [a, b].
  
  :param w: Valor en el intervalo [0, 1].
  :type w: float
  :param a: Límite inferior del intervalo original.
  :type a: float
  :param b: Límite superior del intervalo original.
  :type b: float
  :returns: Valor correspondiente en el intervalo [a, b].
  :rtype: float

.. def:: busquedaDorada(funcion, epsilon, a, b)

  Realiza una búsqueda dorada para minimizar una función en un intervalo dado.
  
  :param funcion: Función unidimensional a minimizar.
  :type funcion: callable
  :param epsilon: Tolerancia para la convergencia.
  :type epsilon: float
  :param a: Límite inferior del intervalo.
  :type a: float
  :param b: Límite superior del intervalo.
  :type b: float
  :returns: Punto en el intervalo [a, b] que minimiza la función.
  :rtype: float

.. def:: gradiente(f, x, deltaX=0.001)

  Calcula el gradiente numérico de una función en un punto dado.
  
  :param f: Función de la cual se quiere calcular el gradiente.
  :type f: callable
  :param x: Lista o tupla con las coordenadas en las que se calcula el gradiente.
  :type x: list or tuple
  :param deltaX: Incremento pequeño para aproximar la derivada (por defecto es 0.001).
  :type deltaX: float
  :returns: Gradiente numérico de la función en el punto dado.
  :rtype: list

.. def:: cauchy(funcion, x0, epsilon1, epsilon2, M, optimizador_univariable)

  Implementa el método de Cauchy para minimizar una función.
  
  :param funcion: Función a minimizar.
  :type funcion: callable
  :param x0: Punto inicial.
  :type x0: list or np.ndarray
  :param epsilon1: Tolerancia para el gradiente.
  :type epsilon1: float
  :param epsilon2: Tolerancia para el cambio en x.
  :type epsilon2: float
  :param M: Número máximo de iteraciones.
  :type M: int
  :param optimizador_univariable: Método de búsqueda unidireccional.
  :type optimizador_univariable: callable
  :returns: Punto que minimiza la función y el valor de la función en ese punto.
  :rtype: np.ndarray, float

**codigo fuente**

.. code-block:: python
 
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

2. Método de Fletcher-Reeves

El método de Fletcher-Reeves es un algoritmo de descenso por gradiente conjugado que utiliza direcciones conjugadas para mejorar la convergencia hacia el mínimo local de una función.

.. def:: gradiente(f, x, deltaX=0.001)

  Calcula el gradiente numérico de una función en un punto dado.
  
  :param f: Función de la cual se quiere calcular el gradiente.
  :type f: callable
  :param x: Lista o tupla con las coordenadas en las que se calcula el gradiente.
  :type x: list or tuple
  :param deltaX: Incremento pequeño para aproximar la derivada (por defecto es 0.001).
  :type deltaX: float
  :returns: Gradiente numérico de la función en el punto dado.
  :rtype: list

.. def:: busqueda_unidireccional(f_lambda, a=0, b=1, tol=1e-5)

  Realiza una búsqueda unidireccional para minimizar una función en un intervalo dado.
  
  :param f_lambda: Función unidimensional a minimizar.
  :type f_lambda: callable
  :param a: Límite inferior del intervalo.
  :type a: float
  :param b: Límite superior del intervalo.
  :type b: float
  :param tol: Tolerancia para la convergencia (por defecto es 1e-5).
  :type tol: float
  :returns: Punto en el intervalo [a, b] que minimiza la función.
  :rtype: float

.. def:: conjugate_gradient_method(f, x0, tol1=1e-5, tol2=1e-5, tol3=1e-5, max_iter=1000)

  Implementa el método del gradiente conjugado para minimizar una función.
  
  :param f: Función a minimizar.
  :type f: callable
  :param x0: Punto inicial.
  :type x0: list or np.ndarray
  :param tol1: Tolerancia para el cambio en x (por defecto es 1e-5).
  :type tol1: float
  :param tol2: Tolerancia relativa para el cambio en x (por defecto es 1e-5).
  :type tol2: float
  :param tol3: Tolerancia para el gradiente (por defecto es 1e-5).
  :type tol3: float
  :param max_iter: Número máximo de iteraciones (por defecto es 1000).
  :type max_iter: int
  :returns: Punto que minimiza la función y el valor de la función en ese punto.
  :rtype: np.ndarray, float

.. code-block:: python
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

3. Método de Newton

El método de Newton es un algoritmo avanzado que utiliza la matriz Hessiana de la función objetivo para calcular la dirección y el tamaño del paso óptimos. Es eficiente pero puede ser sensible a la precisión numérica y requerir evaluaciones exactas de la Hessiana.

.. def:: gradiente(f, x, deltaX=1e-5)

  Calcula el gradiente numérico de una función en un punto dado.
  
  :param f: Función de la cual se quiere calcular el gradiente.
  :type f: callable
  :param x: Lista o tupla con las coordenadas en las que se calcula el gradiente.
  :type x: list or tuple
  :param deltaX: Incremento pequeño para aproximar la derivada (por defecto es 1e-5).
  :type deltaX: float
  :returns: Gradiente numérico de la función en el punto dado.
  :rtype: np.ndarray

.. def::hessian_matrix(f, x, deltaX=1e-5)

  Calcula la matriz Hessiana numérica de una función en un punto dado.
  
  :param f: Función de la cual se quiere calcular la matriz Hessiana.
  :type f: callable
  :param x: Lista o tupla con las coordenadas en las que se calcula la matriz Hessiana.
  :type x: list or tuple
  :param deltaX: Incremento pequeño para aproximar las segundas derivadas (por defecto es 1e-5).
  :type deltaX: float
  :returns: Matriz Hessiana numérica de la función en el punto dado.
  :rtype: np.ndarray

.. def:: newton_method(f, grad_f, x0, tol1=1e-5, tol2=1e-5, tol3=1e-5, max_iter=1000)

  Implementa el método de Newton para minimizar una función.
  
  :param f: Función a minimizar.
  :type f: callable
  :param grad_f: Gradiente de la función.
  :type grad_f: callable
  :param x0: Punto inicial.
  :type x0: list or np.ndarray
  :param tol1: Tolerancia para el gradiente (por defecto es 1e-5).
  :type tol1: float
  :param tol2: Tolerancia para el cambio en x (por defecto es 1e-5).
  :type tol2: float
  :param tol3: Tolerancia para el valor del gradiente en el nuevo punto (por defecto es 1e-5).
  :type tol3: float
  :param max_iter: Número máximo de iteraciones (por defecto es 1000).
  :type max_iter: int
  :returns: Punto que minimiza la función.
  :rtype: np.ndarray


.. code-block:: python
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
        