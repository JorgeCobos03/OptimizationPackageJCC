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

def f1(x):
    return x**2 + 54/x

def f1_derivative(x):
    return 2*x - 54/x**2

def f2(x):
    return x**3 + 2*x - 3

def f2_derivative(x):
    return 3*x**2 + 2

def f3(x):
    return x**4 + x**2 - 33

def f3_derivative(x):
    return 4*x**3 + 2*x

def f4(x):
    return 3*x**4 - 8*x**3 - 6*x**2 + 12*x

def f4_derivative(x):
    return 12*x**3 - 24*x**2 - 12*x + 12

def caja(L):
    return (L * (20 - 2*L) * (10 - 2*L)) * -1

def caja_derivative(L):
    return 200 - 120*L + 12*L**2

def lata_funcion(x):
    return 2 * np.pi * x ** 2 + (500 / x)

def lata_funcion_derivative(x):
    return 4 * np.pi * x - 500 / x**2

# Inicialización de las búsquedas
search_f1 = NewtonRaphsonSearch(f1, f1_derivative, 0.1)
search_f2 = NewtonRaphsonSearch(f2, f2_derivative, -5)
search_f3 = NewtonRaphsonSearch(f3, f3_derivative, -2.5)
search_f4 = NewtonRaphsonSearch(f4, f4_derivative, -1.5)
search_caja = NewtonRaphsonSearch(caja, caja_derivative, 2.5)
search_lata = NewtonRaphsonSearch(lata_funcion, lata_funcion_derivative, 0.1)

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
