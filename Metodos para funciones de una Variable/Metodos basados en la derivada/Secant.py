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

def f1(x):
    return x**2 + 54/x

def f2(x):
    return x**3 + 2*x - 3

def f3(x):
    return x**4 + x**2 - 33

def f4(x):
    return 3*x**4 - 8*x**3 - 6*x**2 + 12*x

def caja(L):
    return (L * (20 - 2*L) * (10 - 2*L)) * -1

def lata_funcion(x):
    return 2 * np.pi * x ** 2 + (500 / x)

# Inicialización de las búsquedas
search_f1 = SecantSearch(f1, 0.1, 1)
search_f2 = SecantSearch(f2, -5, -4)
search_f3 = SecantSearch(f3, -2.5, -2)
search_f4 = SecantSearch(f4, -1.5, -1)
search_caja = SecantSearch(caja, 2, 3)
search_lata = SecantSearch(lata_funcion, 0.1, 1)

# Valores de precisión
precision_values = [0.5, 0.1, 0.01, 0.0001]

# Búsqueda y presentación de resultados
for precision in precision_values:
    print("\nResultados para precisión = {:.4f}:".format(precision))
    print("-" * 70)
    headers = ["Función", "Óptimo (x)", "Valor de la función f(x)"]
    print(f"| {headers[0]:<30} | {headers[1]:<25} | {headers[2]:<25} |")
    print("-" * 70)
    
    # f1
    result_f1 = search_f1.search(precision)
    print(f"| {'f1(x) = x^2 + 54/x':<30} | {result_f1:<25.10f} | {f1(result_f1):<25.10f} |")
    
    # f2
    result_f2 = search_f2.search(precision)
    print(f"| {'f2(x) = x^3 + 2x - 3':<30} | {result_f2:<25.10f} | {f2(result_f2):<25.10f} |")
    
    # f3
    result_f3 = search_f3.search(precision)
    print(f"| {'f3(x) = x^4 + x^2 - 33':<30} | {result_f3:<25.10f} | {f3(result_f3):<25.10f} |")
    
    # f4
    result_f4 = search_f4.search(precision)
    print(f"| {'f4(x) = 3x^4 - 8x^3 - 6x^2 + 12x':<30} | {result_f4:<25.10f} | {f4(result_f4):<25.10f} |")
    
    # lata
    result_lata = search_lata.search(precision)
    print(f"| {'Lata':<30} | {result_lata:<25.10f} | {lata_funcion(result_lata):<25.10f} |")
    
    # caja
    result_caja = search_caja.search(precision)
    print(f"| {'Caja':<30} | {result_caja:<25.10f} | {caja(result_caja):<25.10f} |")
    
    print("-" * 70)
