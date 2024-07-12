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
import numpy as np

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

# Definición de las funciones f1, f2, f3, f4, caja y lata_funcion
def f1(x):
    return x**2 + 54/x

def f2(x):
    return x**3 + 2*x - 3

def f3(x):
    return x**4 + x**2 - 33

def f4(x):
    return 3*x**4 - 8*x**3 - 6*x**2 + 12*x

def caja(L):
    return (L * (20 - 2*L) * (10 - 2*L))*-1

def lata_funcion(x):
    return 2 * np.pi * x ** 2 + (500 / x)

# Inicialización de las búsquedas
search_f1 = FibonacciSearch(f1, 0.1, 10)
search_f2 = FibonacciSearch(f2, -5, 5)
search_f3 = FibonacciSearch(f3, -2.5, 2.5)
search_f4 = FibonacciSearch(f4, -1.5, 3)
search_caja = FibonacciSearch(caja, 2, 3)
search_lata = FibonacciSearch(lata_funcion, 0.1, 10)

# Valores de precisión
precision_values = [0.5, 0.1, 0.01, 0.0001]

# Búsqueda y presentación de resultados en formato de tabla
for precision in precision_values:
    print("\nResultados para precisión = {:.4f}:".format(precision))
    print("-" * 50)
    headers = ["Función", "Óptimo (x)", "Valor de la función f(x)"]
    print(f"| {headers[0]:<30} | {headers[1]:<25} | {headers[2]:<25} |")
    print("-" * 50)
    
    # f1
    result_f1 = search_f1.search(precision)[-1]
    print(f"| {'f1(x) = x^2 + 54/x':<30} | {result_f1:<25.10f} | {f1(result_f1):<25.10f} |")
    
    # f2
    result_f2 = search_f2.search(precision)[-1]
    print(f"| {'f2(x) = x^3 + 2x - 3':<30} | {result_f2:<25.10f} | {f2(result_f2):<25.10f} |")
    
    # f3
    result_f3 = search_f3.search(precision)[-1]
    print(f"| {'f3(x) = x^4 + x^2 - 33':<30} | {result_f3:<25.10f} | {f3(result_f3):<25.10f} |")
    
    # f4
    result_f4 = search_f4.search(precision)[-1]
    print(f"| {'f4(x) = 3x^4 - 8x^3 - 6x^2 + 12x':<30} | {result_f4:<25.10f} | {f4(result_f4):<25.10f} |")
    
    # lata
    result_lata = search_lata.search(precision)[-1]
    print(f"| {'Lata':<30} | {result_lata:<25.10f} | {lata_funcion(result_lata):<25.10f} |")
    
    # caja
    result_caja = search_caja.search(precision)[-1]
    print(f"| {'Caja':<30} | {result_caja:<25.10f} | {caja(result_caja):<25.10f} |")
    
    print("-" * 50)
```
## Instrucciones de Uso
### Instalar dependencias: Asegúrese de tener numpy instalado.

```bash
pip install numpy
```
* Definir funciones: Cree las funciones que desea minimizar.

* Inicializar búsqueda: Cree instancias de FibonacciSearch con las funciones y los intervalos de búsqueda deseados.

* Realizar búsqueda: Llame al método search con el valor de precisión deseado.

* Visualizar resultados: Los resultados se muestran en formato de tabla con los valores óptimos encontrados para cada función.

