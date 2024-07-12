import numpy as np

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

if __name__ == "__main__":
    import numpy as np
    import benchmark_functions as bf
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
