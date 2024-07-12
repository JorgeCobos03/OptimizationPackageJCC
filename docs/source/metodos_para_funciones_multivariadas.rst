Métodos para Funciones Multivariadas
====================================

Este proyecto tambien implementa varios métodos numéricos para encontrar el mínimo de funciones multivariadas en Python. Se incluyen tanto métodos directos como métodos basados en gradientes. A continuación, se describen los métodos y se proporciona un ejemplo de implementación utilizando el algoritmo Hooke-Jeeves.


Métodos Directos
----------------

1. Caminata Aleatoria

El método de caminata aleatoria realiza movimientos aleatorios en el espacio de búsqueda. Es útil para explorar soluciones en problemas donde la función objetivo no es diferenciable o no tiene una estructura regular.

Funciones
^^^^^^^^^

- `random_walk(f_name_or_index, x0, max_iter=1000, epsilon=1e-6, mu=0, sigma=1)`

  Implementa el algoritmo de Random Walk para minimizar la función objetivo.

  **Parámetros**

  - `f_name_or_index` (str or int): El nombre de la función objetivo a minimizar o su índice en `benchmark_functions`.
  - `x0` (list or np.array): La solución inicial.
  - `max_iter` (int, opcional): El número máximo de iteraciones (por defecto es 1000).
  - `epsilon` (float, opcional): Criterio de terminación basado en la tolerancia (por defecto es 1e-6).
  - `mu` (float, opcional): Media de la distribución normal para la generación de pasos aleatorios (por defecto es 0).
  - `sigma` (float, opcional): Desviación estándar de la distribución normal para la generación de pasos aleatorios (por defecto es 1).

  **Retorna**

  - list: La mejor solución encontrada.
  - float: El valor de la función objetivo en la mejor solución.

  **Ejemplo**

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

3. Método de Hooke-Jeeves

El método de Hooke-Jeeves es una técnica de búsqueda directa que explora el espacio de búsqueda mediante pasos incrementales y reducciones adaptativas del tamaño del paso. Es efectivo para encontrar mínimos locales en funciones continuas.

Métodos de Gradiente
--------------------

1. Método de Cauchy

El método de Cauchy utiliza una combinación de descensos por gradiente y pasos de búsqueda lineal para encontrar el mínimo local de una función. Es eficiente pero puede requerir ajustes en el tamaño de paso.

2. Método de Fletcher-Reeves

El método de Fletcher-Reeves es un algoritmo de descenso por gradiente conjugado que utiliza direcciones conjugadas para mejorar la convergencia hacia el mínimo local de una función.

3. Método de Newton

El método de Newton es un algoritmo avanzado que utiliza la matriz Hessiana de la función objetivo para calcular la dirección y el tamaño del paso óptimos. Es eficiente pero puede ser sensible a la precisión numérica y requerir evaluaciones exactas de la Hessiana.

