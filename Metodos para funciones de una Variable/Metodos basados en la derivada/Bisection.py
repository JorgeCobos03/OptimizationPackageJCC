import numpy as np

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
search_f1 = BisectionSearch(f1, 0.1, 10)
search_f2 = BisectionSearch(f2, -5, 5)
search_f3 = BisectionSearch(f3, -2.5, 2.5)
search_f4 = BisectionSearch(f4, -1.5, 3)
search_caja = BisectionSearch(caja, 2, 3)
search_lata = BisectionSearch(lata_funcion, 0.1, 10)

# Valores de precisión
precision_values = [0.5, 0.1, 0.01, 0.0001]

# Búsqueda y presentación de resultados
for precision in precision_values:
    print("\nResultados para precisión = {:.4f}:".format(precision))
    print("-" * 50)
    headers = ["Función", "Óptimo (x)", "Valor de la función f(x)"]
    print(f"| {headers[0]:<30} | {headers[1]:<25} | {headers[2]:<25} |")
    print("-" * 50)
    
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
    
    print("-" * 50)
