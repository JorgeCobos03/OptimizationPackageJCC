import numpy as np

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

if __name__ == "__main__":
    import numpy as np
    import benchmark_functions as bf
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
        search_instance = SecantSearch(func, lower_bound, upper_bound)
        precision = 0.0001  # Definir la precisión deseada para la búsqueda
        search_results[name] = search_instance.search(precision)

    # Mostrar resultados
    print("\nResultados de la búsqueda con método de Bisección:")
    print("-" * 50)
    for name, result in search_results.items():
        print(f"{name}: Óptimo (x) = {result:.10f} | Valor de la función f(x) = {benchmark_functions[name](result):.10f}")
