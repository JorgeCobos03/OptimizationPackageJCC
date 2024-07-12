import numpy as np

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
        search_instance = IntervalHalvingSearch(func, lower_bound, upper_bound)
        precision = 0.0001  # Definir la precisión deseada para la búsqueda
        search_results[name] = search_instance.search(precision)

    # Mostrar resultados
    print("\nResultados de la búsqueda con método de Interval Halving:")
    print("-" * 50)
    for name, result in search_results.items():
        print(f"{name}: Óptimo (x) = {result:.10f} | Valor de la función f(x) = {benchmark_functions[name](result):.10f}")
