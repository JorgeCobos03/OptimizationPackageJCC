uso
=====

.. _Instalación:

Instalación
------------

Este paquete puede ser instalado fácilmente utilizando `pip`, el gestor de paquetes de Python. A continuación se detallan los pasos necesarios para instalar `OptimizationPackageJCC` desde el índice de prueba de PyPI.

Primero, asegúrate de tener `pip` instalado en tu sistema. Si no lo tienes, puedes instalarlo siguiendo las instrucciones en la documentación oficial de [pip](https://pip.pypa.io/en/stable/installation/).

Para instalar el paquete `OptimizationPackageJCC`, ejecuta el siguiente comando en tu terminal:

.. code-block:: bash

    pip install -i https://test.pypi.org/simple/ OptimizationPackageJCC==0.0.1

Este comando instalará la versión 0.0.1 del paquete desde el índice de prueba de PyPI.

**Nota**: El índice de prueba de PyPI (`test.pypi.org`) es una versión separada de PyPI que se utiliza para probar la distribución de paquetes. No se recomienda utilizar este índice para instalaciones en entornos de producción. Para versiones de producción, asegúrate de instalar desde el índice principal de PyPI (`pypi.org`).

Si prefieres instalar desde el índice principal de PyPI, usa el siguiente comando (asegúrate de que el paquete esté publicado en PyPI):

.. code-block:: bash

    pip install optimizationpackagejcc

**Requisitos Previos**

Asegúrate de tener las siguientes dependencias instaladas antes de proceder con la instalación del paquete:
- Python 3.6 o superior
- pip

Para verificar la versión de Python y pip instaladas en tu sistema, puedes usar los siguientes comandos:

.. code-block:: bash

    python --version
    pip --version

**Solución de Problemas**

Si encuentras problemas durante la instalación, aquí tienes algunos pasos que puedes seguir para resolverlos:

1. **Actualizar pip**: Asegúrate de que `pip` esté actualizado a la última versión.
   
   .. code-block:: bash

       pip install --upgrade pip

2. **Entornos Virtuales**: Se recomienda utilizar un entorno virtual para gestionar las dependencias del proyecto. Puedes crear y activar un entorno virtual usando `venv`:

   .. code-block:: bash

       python -m venv myenv
       source myenv/bin/activate  # En Windows usa `myenv\Scripts\activate`

3. **Dependencias Faltantes**: Si faltan dependencias, `pip` debería manejarlas automáticamente. Si no, instala las dependencias manualmente listadas en `requirements.txt` o según la documentación del paquete.

Si después de seguir estos pasos aún tienes problemas, por favor abre un issue en el repositorio del proyecto en GitHub para obtener asistencia.

Métodos para Funciones de una Variable
---------------------------------------

Este proyecto implementa varios métodos numéricos para encontrar
el mínimo de funciones de una variable en Python. Se incluyen tanto 
métodos de eliminación de regiones como métodos basados en la derivada. 
A continuación, se describen los métodos y se proporciona un ejemplo de 
implementación utilizando el método de búsqueda de Fibonacci.

Métodos de Eliminación de Regiones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Los Métodos de Eliminación de Regiones son técnicas utilizadas 
para encontrar el mínimo de una función univariada o multivariada 
al dividir iterativamente el dominio de la función en subintervalos 
más pequeños. Estos métodos son eficaces para reducir gradualmente 
el espacio de búsqueda hasta localizar la región que contiene el 
mínimo deseado. Para este caso es univariada con ejemplos incluyen 
el Método de División de Intervalos por la Mitad y la Búsqueda de 
Fibonacci, que optimizan la búsqueda reduciendo el número de 
evaluaciones de la función en cada paso.

Método de División de Intervalos por la Mitad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

El método de división de intervalos por la mitad 
consiste en dividir el intervalo de búsqueda en dos 
subintervalos y evaluar la función en los puntos medios 
de estos subintervalos. Se selecciona el subintervalo 
que contiene el mínimo y se repite el proceso hasta 
alcanzar la precisión deseada.

Búsqueda de Fibonacci
^^^^^^^^^^^^^^^^^^^^

La búsqueda de Fibonacci es otro método de optimización
que utiliza los números de Fibonacci para dividir el intervalo
de búsqueda. Es eficiente en términos de evaluaciones de la función
y converge más rápido que el método de división de intervalos por la mitad.

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


Método de la Sección Dorada
^^^^^^^^^^^^^^^^^^^^^^^^^^^

El método de la sección dorada es un caso especial del método
de división de intervalos que utiliza la proporción áurea para
elegir los puntos de evaluación. Esto minimiza el número de evaluaciones necesarias.

Métodos Basados en la Derivada
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Los Métodos Basados en la Derivada son técnicas utilizadas para encontrar 
mínimos de funciones mediante el análisis de sus derivadas. Estos métodos 
son eficaces cuando se dispone de información sobre la pendiente de la función 
en puntos específicos. Ejemplos incluyen el Método de Newton-Raphson, que utiliza 
derivadas para iterar hacia mínimos locales, el Método de Bisección, que encuentra 
raíces de funciones univariadas para localizar mínimos en derivadas, y el Método de 
la Secante, una variante del método de Newton-Raphson que no requiere la segunda derivada.

Método de Newton-Raphson
^^^^^^^^^^^^^^^^^^^^^^^^

El método de Newton-Raphson es un método iterativo para encontrar
raíces de una función. Se puede adaptar para encontrar mínimos al
buscar puntos donde la derivada de la función es cero.
Utiliza derivadas de la función para encontrar sus raíces, adaptado para encontrar mínimos.

Método de Bisección
^^^^^^^^^^^^^^^^^^^

El método de bisección es un método de búsqueda de raíces que divide
el intervalo de búsqueda en dos partes iguales y selecciona el 
subintervalo que contiene una raíz. Se puede adaptar para encontrar
mínimos buscando cambios de signo en la derivada de la función.

Método de la Secante
^^^^^^^^^^^^^^^^^^^^

El método de la secante es similar al método de Newton-Raphson pero
no requiere el cálculo de la derivada. En su lugar, utiliza una secante
a la curva para aproximar la raíz.


Métodos para Funciones Multivariadas
-------------------------------------

Métodos Directos
~~~~~~~~~~~~~~~~

Caminata Aleatoria
^^^^^^^^^^^^^^^^^^

Explora el espacio de búsqueda de manera aleatoria para encontrar un mínimo.

Método de Nelder y Mead (Simplex)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Utiliza un simplejo de puntos para iterativamente aproximarse al mínimo.

Método de Hooke-Jeeves
^^^^^^^^^^^^^^^^^^^^^^

Un algoritmo de búsqueda directa que combina exploración y patrones de búsqueda.

Métodos de Gradiente
~~~~~~~~~~~~~~~~~~~~

Método de Cauchy
^^^^^^^^^^^^^^^^

Utiliza el gradiente para iterar hacia el mínimo siguiendo la dirección de mayor descenso.

Método de Fletcher-Reeves
^^^^^^^^^^^^^^^^^^^^^^^^^

Una técnica de gradiente conjugado que mejora la eficiencia del método de Cauchy.

Método de Newton
^^^^^^^^^^^^^^^^

Utiliza la información de la segunda derivada (Hessiana) para encontrar el mínimo más rápidamente.
