Métodos para Funciones Univariadas
======================================

Este proyecto implementa varios métodos numéricos para encontrar
el mínimo de funciones de una variable en Python. Se incluyen tanto 
métodos de eliminación de regiones como métodos basados en la derivada. 
A continuación, se describen los métodos y se proporciona un ejemplo de 
implementación utilizando el método de búsqueda de Fibonacci.

Métodos de Eliminación de Regiones
----------------------------------

Los Métodos de Eliminación de Regiones son técnicas utilizadas 
para encontrar el mínimo de una función univariada o multivariada 
al dividir iterativamente el dominio de la función en subintervalos 
más pequeños. Estos métodos son eficaces para reducir gradualmente 
el espacio de búsqueda hasta localizar la región que contiene el 
mínimo deseado. Para este caso es univariada con ejemplos incluyen 
el Método de División de Intervalos por la Mitad y la Búsqueda de 
Fibonacci, que optimizan la búsqueda reduciendo el número de 
evaluaciones de la función en cada paso.

1. Método de División de Intervalos por la Mitad

El método de división de intervalos por la mitad 
consiste en dividir el intervalo de búsqueda en dos 
subintervalos y evaluar la función en los puntos medios 
de estos subintervalos. Se selecciona el subintervalo 
que contiene el mínimo y se repite el proceso hasta 
alcanzar la precisión deseada.

- Expliación

Clase que implementa el método de búsqueda de intervalos mediante la división por la mitad.

.. class:: IntervalHalvingSearch(func, lower_bound, upper_bound)

    Inicializa la clase IntervalHalvingSearch.

    :param func: La función a minimizar.
    :type func: function
    :param lower_bound: El límite inferior del intervalo de búsqueda.
    :type lower_bound: float
    :param upper_bound: El límite superior del intervalo de búsqueda.
    :type upper_bound: float

    .. method:: search(precision)

        Realiza la búsqueda de intervalos mediante la división por la mitad.

        :param precision: La precisión deseada para la búsqueda.
        :type precision: float
        :returns: El punto óptimo encontrado.
        :rtype: float

- Ejemplo de Uso

.. code-block:: python

    if __name__ == "__main__":
        import numpy as np
        import benchmark_functions as bf

        class IntervalHalvingSearch:
            """
            Clase que implementa el método de búsqueda de intervalos mediante la división por la mitad.

            Attributes
            ----------
            func : function
                La función a minimizar.
            lower_bound : float
                El límite inferior del intervalo de búsqueda.
            upper_bound : float
                El límite superior del intervalo de búsqueda.
            """

            def __init__(self, func, lower_bound, upper_bound):
                """
                Inicializa la clase IntervalHalvingSearch.

                Parameters
                ----------
                func : function
                    La función a minimizar.
                lower_bound : float
                    El límite inferior del intervalo de búsqueda.
                upper_bound : float
                    El límite superior del intervalo de búsqueda.
                """
                self.func = func
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound

            def search(self, precision):
                """
                Realiza la búsqueda de intervalos mediante la división por la mitad.

                Parameters
                ----------
                precision : float
                    La precisión deseada para la búsqueda.

                Returns
                -------
                float
                    El punto óptimo encontrado.
                """
                a = self.lower_bound
                b = self.upper_bound
                delta = precision / 2

                while abs(b - a) > precision:
                    x1 = a + delta
                    x2 = b - delta
                    if self.func(x1) < self.func(x2):
                        b = x2
                    else:
                        a = x1
                return (a + b) / 2

        benchmark_functions = {
            "f1": bf.f1,
            "f2": bf.f2,
            "f3": bf.f3,
            "f4": bf.f4,
            "caja": bf.caja,
            "lata_funcion": bf.lata_funcion
        }

        search_results = {}

        for name, func in benchmark_functions.items():
            lower_bound = 0.1  # Definir límite inferior según la función
            upper_bound = 10.0  # Definir límite superior según la función
            search_instance = IntervalHalvingSearch(func, lower_bound, upper_bound)
            precision = 0.0001  # Definir la precisión deseada para la búsqueda
            search_results[name] = search_instance.search(precision)

        # Mostrar resultados
        print("\nResultados de la búsqueda con método de Interval Halving:")
        print("-" * 50)
        for name, result in search_results.items():
            print(f"{name}: Óptimo (x) = {result:.10f} | Valor de la función f(x) = {benchmark_functions[name](result):.10f}")


2. Búsqueda de Fibonacci

La búsqueda de Fibonacci es otro método de optimización
que utiliza los números de Fibonacci para dividir el intervalo
de búsqueda. Es eficiente en términos de evaluaciones de la función
y converge más rápido que el método de división de intervalos por la mitad.

- Explicación

Clase que implementa el método de búsqueda de Fibonacci para encontrar el mínimo de una función.

.. class:: FibonacciSearch(func, lower_bound, upper_bound)

    Inicializa la clase FibonacciSearch.

    :param func: La función a minimizar.
    :type func: function
    :param lower_bound: El límite inferior del intervalo de búsqueda.
    :type lower_bound: float
    :param upper_bound: El límite superior del intervalo de búsqueda.
    :type upper_bound: float

    .. method:: fibonacci(n)

        Calcula el enésimo número de Fibonacci.

        :param n: El índice del número de Fibonacci a calcular.
        :type n: int
        :returns: El enésimo número de Fibonacci.
        :rtype: int

    .. method:: search(precision)

        Realiza la búsqueda de Fibonacci para encontrar el mínimo de la función.

        :param precision: La precisión deseada para la búsqueda.
        :type precision: float
        :returns: El valor del punto medio del intervalo de búsqueda después de cada iteración.
        :rtype: float

- Ejemplo de Uso

.. code-block:: python

    if __name__ == "__main__":
        import numpy as np
        import benchmark_functions as bf

        class FibonacciSearch:
            """
            Clase que implementa el método de búsqueda de Fibonacci para encontrar el mínimo de una función.

            Attributes
            ----------
            func : function
                La función a minimizar.
            lower_bound : float
                El límite inferior del intervalo de búsqueda.
            upper_bound : float
                El límite superior del intervalo de búsqueda.
            """

            def __init__(self, func, lower_bound, upper_bound):
                """
                Inicializa la clase FibonacciSearch.

                Parameters
                ----------
                func : function
                    La función a minimizar.
                lower_bound : float
                    El límite inferior del intervalo de búsqueda.
                upper_bound : float
                    El límite superior del intervalo de búsqueda.
                """
                self.func = func
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound

            def fibonacci(self, n):
                """
                Calcula el enésimo número de Fibonacci.

                Parameters
                ----------
                n : int
                    El índice del número de Fibonacci a calcular.

                Returns
                -------
                int
                    El enésimo número de Fibonacci.
                """
                if n <= 1:
                    return n
                else:
                    return self.fibonacci(n-1) + self.fibonacci(n-2)

            def search(self, precision):
                """
                Realiza la búsqueda de Fibonacci para encontrar el mínimo de la función.

                Parameters
                ----------
                precision : float
                    La precisión deseada para la búsqueda.

                Returns
                -------
                float
                    El valor del punto medio del intervalo de búsqueda después de cada iteración.
                """
                iterations = []
                n = 0
                while self.fibonacci(n) < (self.upper_bound - self.lower_bound) / precision:
                    n += 1
                fib_n = self.fibonacci(n)
                x1 = self.lower_bound + (self.upper_bound - self.lower_bound) * self.fibonacci(n-2) / fib_n
                x2 = self.lower_bound + (self.upper_bound - self.lower_bound) * self.fibonacci(n-1) / fib_n

                for _ in range(n-2):  # Usamos n-2 porque n-1 es la última iteración
                    if self.func(x1) < self.func(x2):
                        self.upper_bound = x2
                        x2 = x1
                        x1 = self.lower_bound + (self.upper_bound - self.lower_bound) * self.fibonacci(n-3) / fib_n
                    else:
                        self.lower_bound = x1
                        x1 = x2
                        x2 = self.lower_bound + (self.upper_bound - self.lower_bound) * self.fibonacci(n-2) / fib_n
                    iterations.append((self.lower_bound + self.upper_bound) / 2)

                # Comparar x1 y x2 en la última iteración
                if self.func(x1) < self.func(x2):
                    iterations.append(x1)
                else:
                    iterations.append(x2)

                return iterations

        benchmark_functions = {
            "f1": bf.f1,
            "f2": bf.f2,
            "f3": bf.f3,
            "f4": bf.f4,
            "caja": bf.caja,
            "lata_funcion": bf.lata_funcion
        }

        search_results = {}

        for name, func in benchmark_functions.items():
            lower_bound = 0.1  # Definir límite inferior según la función
            upper_bound = 10.0  # Definir límite superior según la función
            search_instance = FibonacciSearch(func, lower_bound, upper_bound)
            precision = 0.0001  # Definir la precisión deseada para la búsqueda
            search_results[name] = search_instance.search(precision)[-1]

        # Mostrar resultados
        print("\nResultados de la búsqueda con método de Fibonacci:")
        print("-" * 50)
        for name, result in search_results.items():
            print(f"{name}: Óptimo (x) = {result:.10f} | Valor de la función f(x) = {benchmark_functions[name](result):.10f}")


3. Método de la Sección Dorada

El método de la sección dorada es un caso especial del método
de división de intervalos que utiliza la proporción áurea para
elegir los puntos de evaluación. Esto minimiza el número de evaluaciones necesarias.

- Explicación

Clase que implementa el método de búsqueda de la sección áurea.

.. class:: GoldenSectionSearch(func, lower_bound, upper_bound)

    Inicializa la clase GoldenSectionSearch.

    :param func: La función a minimizar.
    :type func: function
    :param lower_bound: El límite inferior del intervalo de búsqueda.
    :type lower_bound: float
    :param upper_bound: El límite superior del intervalo de búsqueda.
    :type upper_bound: float

    .. method:: search(precision)

        Realiza la búsqueda mediante la sección áurea.

        :param precision: La precisión deseada para la búsqueda.
        :type precision: float
        :returns: El punto óptimo encontrado.
        :rtype: float

- Ejemplo de Uso

.. code-block:: python

    if __name__ == "__main__":
        import numpy as np
        import benchmark_functions as bf

        class GoldenSectionSearch:
            """
            Clase que implementa el método de búsqueda de la sección áurea.

            Attributes
            ----------
            func : function
                La función a minimizar.
            lower_bound : float
                El límite inferior del intervalo de búsqueda.
            upper_bound : float
                El límite superior del intervalo de búsqueda.
            """

            def __init__(self, func, lower_bound, upper_bound):
                """
                Inicializa la clase GoldenSectionSearch.

                Parameters
                ----------
                func : function
                    La función a minimizar.
                lower_bound : float
                    El límite inferior del intervalo de búsqueda.
                upper_bound : float
                    El límite superior del intervalo de búsqueda.
                """
                self.func = func
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound

            def search(self, precision):
                """
                Realiza la búsqueda mediante la sección áurea.

                Parameters
                ----------
                precision : float
                    La precisión deseada para la búsqueda.

                Returns
                -------
                float
                    El punto óptimo encontrado.
                """
                gr = (np.sqrt(5) - 1) / 2
                a = self.lower_bound
                b = self.upper_bound

                while abs(b - a) > precision:
                    x1 = b - gr * (b - a)
                    x2 = a + gr * (b - a)
                    if self.func(x1) < self.func(x2):
                        b = x2
                    else:
                        a = x1
                return (a + b) / 2

        benchmark_functions = {
            "f1": bf.f1,
            "f2": bf.f2,
            "f3": bf.f3,
            "f4": bf.f4,
            "caja": bf.caja,
            "lata_funcion": bf.lata_funcion
        }

        search_results = {}

        for name, func in benchmark_functions.items():
            lower_bound = 0.1  # Definir límite inferior según la función
            upper_bound = 10.0  # Definir límite superior según la función
            search_instance = GoldenSectionSearch(func, lower_bound, upper_bound)
            precision = 0.0001  # Definir la precisión deseada para la búsqueda
            search_results[name] = search_instance.search(precision)

        # Mostrar resultados
        print("\nResultados de la búsqueda con método de Sección Áurea:")
        print("-" * 50)
        for name, result in search_results.items():
            print(f"{name}: Óptimo (x) = {result:.10f} | Valor de la función f(x) = {benchmark_functions[name](result):.10f}")

Métodos Basados en la Derivada
------------------------------

Los Métodos Basados en la Derivada son técnicas utilizadas para encontrar 
mínimos de funciones mediante el análisis de sus derivadas. Estos métodos 
son eficaces cuando se dispone de información sobre la pendiente de la función 
en puntos específicos. Ejemplos incluyen el Método de Newton-Raphson, que utiliza 
derivadas para iterar hacia mínimos locales, el Método de Bisección, que encuentra 
raíces de funciones univariadas para localizar mínimos en derivadas, y el Método de 
la Secante, una variante del método de Newton-Raphson que no requiere la segunda derivada.

1. Método de Newton-Raphson

El método de Newton-Raphson es un método iterativo para encontrar
raíces de una función. Se puede adaptar para encontrar mínimos al
buscar puntos donde la derivada de la función es cero.
Utiliza derivadas de la función para encontrar sus raíces, adaptado para encontrar mínimos.

- Explicación

Clase que implementa el método de búsqueda de Newton-Raphson.

.. class:: NewtonRaphsonSearch(func, derivative, initial_guess)

    Inicializa la clase NewtonRaphsonSearch.

    :param func: La función a minimizar.
    :type func: function
    :param derivative: La derivada de la función a minimizar.
    :type derivative: function
    :param initial_guess: La estimación inicial para la búsqueda.
    :type initial_guess: float

    .. method:: search(precision, max_iter=100)

        Realiza la búsqueda mediante el método de Newton-Raphson.

        :param precision: La precisión deseada para la búsqueda.
        :type precision: float
        :param max_iter: El número máximo de iteraciones (por defecto es 100).
        :type max_iter: int, optional
        :returns: El punto óptimo encontrado.
        :rtype: float

- Ejemplo de Uso


.. code-block:: python

    if __name__ == "__main__":
        import numpy as np
        import benchmark_functions as bf

        class NewtonRaphsonSearch:
            """
            Clase que implementa el método de búsqueda de Newton-Raphson.

            Attributes
            ----------
            func : function
                La función a minimizar.
            derivative : function
                La derivada de la función a minimizar.
            initial_guess : float
                La estimación inicial para la búsqueda.
            """

            def __init__(self, func, derivative, initial_guess):
                """
                Inicializa la clase NewtonRaphsonSearch.

                Parameters
                ----------
                func : function
                    La función a minimizar.
                derivative : function
                    La derivada de la función a minimizar.
                initial_guess : float
                    La estimación inicial para la búsqueda.
                """
                self.func = func
                self.derivative = derivative
                self.initial_guess = initial_guess

            def search(self, precision, max_iter=100):
                """
                Realiza la búsqueda mediante el método de Newton-Raphson.

                Parameters
                ----------
                precision : float
                    La precisión deseada para la búsqueda.
                max_iter : int, optional
                    El número máximo de iteraciones (por defecto es 100).

                Returns
                -------
                float
                    El punto óptimo encontrado.
                """
                x = self.initial_guess

                for _ in range(max_iter):
                    x_next = x - self.func(x) / self.derivative(x)
                    if abs(x_next - x) < precision:
                        return x_next
                    x = x_next

                return x

        benchmark_functions = {
            "f1": bf.f1,
            "f1_derivative": bf.f1_derivative,
            "f2": bf.f2,
            "f2_derivative": bf.f2_derivative,
            "f3": bf.f3,
            "f3_derivative": bf.f3_derivative,
            "f4": bf.f4,
            "f4_derivative": bf.f4_derivative,
            "caja": bf.caja,
            "caja_derivative": bf.caja_derivative,
            "lata_funcion": bf.lata_funcion,
            "lata_funcion_derivative": bf.lata_funcion_derivative
        }

        search_results = {}

        for name, func in benchmark_functions.items():
            if "derivative" in name:
                continue  # Saltar las funciones de derivadas aquí
            initial_guess = 0.1  # Estimación inicial para cada función de benchmark
            derivative_func = benchmark_functions[name + "_derivative"]  # Obtener la función derivada correspondiente
            search_instance = NewtonRaphsonSearch(func, derivative_func, initial_guess)
            precision = 0.0001  # Definir la precisión deseada para la búsqueda
            search_results[name] = search_instance.search(precision)

        # Mostrar resultados
        print("\nResultados de la búsqueda con método de Newton-Raphson:")
        print("-" * 50)
        for name, result in search_results.items():
            print(f"{name}: Óptimo (x) = {result:.10f} | Valor de la función f(x) = {benchmark_functions[name](result):.10f}")

2. Método de Bisección

El método de bisección es un método de búsqueda de raíces que divide
el intervalo de búsqueda en dos partes iguales y selecciona el 
subintervalo que contiene una raíz. Se puede adaptar para encontrar
mínimos buscando cambios de signo en la derivada de la función.

- Explicación

Clase que implementa el método de búsqueda por bisección.

.. class:: BisectionSearch(func, lower_bound, upper_bound)

    Inicializa la clase BisectionSearch.

    :param func: La función a minimizar.
    :type func: function
    :param lower_bound: El límite inferior del intervalo de búsqueda.
    :type lower_bound: float
    :param upper_bound: El límite superior del intervalo de búsqueda.
    :type upper_bound: float

    .. method:: search(precision)

        Realiza la búsqueda mediante el método de bisección.

        :param precision: La precisión deseada para la búsqueda.
        :type precision: float
        :returns: El punto óptimo encontrado.
        :rtype: float

- Ejemplo de Uso


.. code-block:: python

    if __name__ == "__main__":
        import numpy as np
        import benchmark_functions as bf

        class BisectionSearch:
            """
            Clase que implementa el método de búsqueda por bisección.

            Attributes
            ----------
            func : function
                La función a minimizar.
            lower_bound : float
                El límite inferior del intervalo de búsqueda.
            upper_bound : float
                El límite superior del intervalo de búsqueda.
            """

            def __init__(self, func, lower_bound, upper_bound):
                """
                Inicializa la clase BisectionSearch.

                Parameters
                ----------
                func : function
                    La función a minimizar.
                lower_bound : float
                    El límite inferior del intervalo de búsqueda.
                upper_bound : float
                    El límite superior del intervalo de búsqueda.
                """
                self.func = func
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound

            def search(self, precision):
                """
                Realiza la búsqueda mediante el método de bisección.

                Parameters
                ----------
                precision : float
                    La precisión deseada para la búsqueda.

                Returns
                -------
                float
                    El punto óptimo encontrado.
                """
                a = self.lower_bound
                b = self.upper_bound

                while abs(b - a) > precision:
                    c = (a + b) / 2
                    if self.func(c) == 0:
                        return c
                    elif self.func(a) * self.func(c) < 0:
                        b = c
                    else:
                        a = c

                return (a + b) / 2

        benchmark_functions = {
            "f1": bf.f1,
            "f2": bf.f2,
            "f3": bf.f3,
            "f4": bf.f4,
            "caja": bf.caja,
            "lata_funcion": bf.lata_funcion
        }

        # Inicialización de las búsquedas para cada función de benchmark
        search_results = {}

        for name, func in benchmark_functions.items():
            lower_bound = 0.1  # Definir límite inferior según la función
            upper_bound = 10.0  # Definir límite superior según la función
            search_instance = BisectionSearch(func, lower_bound, upper_bound)
            precision = 0.0001  # Definir la precisión deseada para la búsqueda
            search_results[name] = search_instance.search(precision)

        # Mostrar resultados
        print("\nResultados de la búsqueda con método de Bisección:")
        print("-" * 50)
        for name, result in search_results.items():
            print(f"{name}: Óptimo (x) = {result:.10f} | Valor de la función f(x) = {benchmark_functions[name](result):.10f}")

3. Método de la Secante


El método de la secante es similar al método de Newton-Raphson pero
no requiere el cálculo de la derivada. En su lugar, utiliza una secante
a la curva para aproximar la raíz.

- Expliación

Clase que implementa el método de búsqueda secante.

.. class:: SecantSearch(func, initial_guess1, initial_guess2)

    Inicializa la clase SecantSearch.

    :param func: La función a minimizar.
    :type func: function
    :param initial_guess1: Primer valor de conjetura inicial.
    :type initial_guess1: float
    :param initial_guess2: Segundo valor de conjetura inicial.
    :type initial_guess2: float

    .. method:: search(precision, max_iter=100)

        Realiza la búsqueda mediante el método secante.

        :param precision: La precisión deseada para la búsqueda.
        :type precision: float
        :param max_iter: Número máximo de iteraciones (por defecto es 100).
        :type max_iter: int, optional
        :returns: El punto óptimo encontrado.
        :rtype: float

- Ejemplo de Uso


.. code-block:: python

    if __name__ == "__main__":
        import numpy as np
        import benchmark_functions as bf

        class SecantSearch:
            """
            Clase que implementa el método de búsqueda secante.

            Attributes
            ----------
            func : function
                La función a minimizar.
            initial_guess1 : float
                Primer valor de conjetura inicial.
            initial_guess2 : float
                Segundo valor de conjetura inicial.
            """

            def __init__(self, func, initial_guess1, initial_guess2):
                """
                Inicializa la clase SecantSearch.

                Parameters
                ----------
                func : function
                    La función a minimizar.
                initial_guess1 : float
                    Primer valor de conjetura inicial.
                initial_guess2 : float
                    Segundo valor de conjetura inicial.
                """
                self.func = func
                self.initial_guess1 = initial_guess1
                self.initial_guess2 = initial_guess2

            def search(self, precision, max_iter=100):
                """
                Realiza la búsqueda mediante el método secante.

                Parameters
                ----------
                precision : float
                    La precisión deseada para la búsqueda.
                max_iter : int, optional
                    Número máximo de iteraciones (por defecto es 100).

                Returns
                -------
                float
                    El punto óptimo encontrado.
                """
                x0 = self.initial_guess1
                x1 = self.initial_guess2

                for _ in range(max_iter):
                    x_next = x1 - (self.func(x1) * (x1 - x0)) / (self.func(x1) - self.func(x0))
                    if abs(x_next - x1) < precision:
                        return x_next
                    x0 = x1
                    x1 = x_next

                return x1

        benchmark_functions = {
            "f1": bf.f1,
            "f2": bf.f2,
            "f3": bf.f3,
            "f4": bf.f4,
            "caja": bf.caja,
            "lata_funcion": bf.lata_funcion
        }

        search_results = {}

        for name, func in benchmark_functions.items():
            initial_guess1 = 0.1  # Definir primer valor de conjetura inicial según la función
            initial_guess2 = 1.0  # Definir segundo valor de conjetura inicial según la función
            search_instance = SecantSearch(func, initial_guess1, initial_guess2)
            precision = 0.0001  # Definir la precisión deseada para la búsqueda
            search_results[name] = search_instance.search(precision)

        # Mostrar resultados
        print("\nResultados de la búsqueda con método de la Secante:")
        print("-" * 50)
        for name, result in search_results.items():
            print(f"{name}: Óptimo (x) = {result:.10f} | Valor de la función f(x) = {benchmark_functions[name](result):.10f}")