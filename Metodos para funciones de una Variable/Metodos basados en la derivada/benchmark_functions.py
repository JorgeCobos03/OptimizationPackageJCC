import numpy as np

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