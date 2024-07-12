
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
