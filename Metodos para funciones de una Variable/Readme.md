# Métodos para Funciones de una Variable

Este proyecto implementa varios métodos numéricos para encontrar el mínimo de funciones de una variable en Python. Se incluyen tanto métodos de eliminación de regiones como métodos basados en la derivada. A continuación, se describen los métodos y se proporciona un ejemplo de implementación utilizando el método de búsqueda de Fibonacci.

## Métodos de Eliminación de Regiones

* Método de División de Intervalos por la Mitad

El método de división de intervalos por la mitad consiste en dividir el intervalo de búsqueda en dos subintervalos y evaluar la función en los puntos medios de estos subintervalos. Se selecciona el subintervalo que contiene el mínimo y se repite el proceso hasta alcanzar la precisión deseada.

* Búsqueda de Fibonacci

La búsqueda de Fibonacci es un método de optimización que utiliza los números de Fibonacci para dividir el intervalo de búsqueda. Es eficiente en términos de evaluaciones de la función y converge más rápido que el método de división de intervalos por la mitad.

* Método de la Sección Dorada

El método de la sección dorada es un caso especial del método de división de intervalos que utiliza la proporción áurea para elegir los puntos de evaluación. Esto minimiza el número de evaluaciones necesarias.

## Métodos Basados en la Derivada

* Método de Newton-Raphson

El método de Newton-Raphson es un método iterativo para encontrar raíces de una función. Se puede adaptar para encontrar mínimos al buscar puntos donde la derivada de la función es cero.

* Método de Bisección

El método de bisección es un método de búsqueda de raíces que divide el intervalo de búsqueda en dos partes iguales y selecciona el subintervalo que contiene una raíz. Se puede adaptar para encontrar mínimos buscando cambios de signo en la derivada de la función.

* Método de la Secante

El método de la secante es similar al método de Newton-Raphson pero no requiere el cálculo de la derivada. En su lugar, utiliza una secante a la curva para aproximar la raíz.

## Ejemplo de Implementación: Búsqueda de Fibonacci

```python

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

```
## Instrucciones de Uso
1. Instalar dependencias: Asegúrese de tener numpy instalado, o instalelo con la siguiente linea en la terminal:

```bash
pip install numpy
```
2. Definir funciones: Cree las funciones que desea minimizar como las de de ejemplo en "benchmark_functions.py".

3. Inicializar búsqueda: Cree instancias con las funciones y los intervalos de búsqueda deseados.

4. Realizar búsqueda: Llame al método search con el valor de precisión deseado.

5. Visualizar resultados: Los resultados se muestran en formato de tabla con los valores óptimos encontrados para cada función.

