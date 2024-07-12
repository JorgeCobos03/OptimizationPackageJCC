import numpy as np

def rastrigin(X):
    return 10 * len(X) + sum([(x**2 - 10 * np.cos(2 * np.pi * x)) for x in X])

def ackley(X):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (X[0]**2 + X[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * X[0]) + np.cos(2 * np.pi * X[1]))) + 20 + np.e

def sphere(X):
    return X[0]**2 + X[1]**2

def rosenbrock(X):
    return (1 - X[0])**2 + 100 * (X[1] - X[0]**2)**2

def beale(X):
    return (1.5 - X[0] + X[0] * X[1])**2 + (2.25 - X[0] + X[0] * X[1]**2)**2 + (2.625 - X[0] + X[0] * X[1]**3)**2

def goldstein_price(X):
    return (1 + ((X[0] + X[1] + 1)**2) * (19 - 14*X[0] + 3*(X[0]**2) - 14*X[1] + 6*X[0]*X[1] + 3*(X[1]**2))) * (30 + ((2*X[0] - 3*X[1])**2) * (18 - 32*X[0] + 12*(X[0]**2) + 48*X[1] - 36*X[0]*X[1] + 27*(X[1]**2)))

def booth(X):
    return (X[0] + 2*X[1] - 7)**2 + (2*X[0] + X[1] - 5)**2

def bukin6(X):
    return 100 * np.sqrt(np.abs(X[1] - 0.01*X[0]**2)) + 0.01 * np.abs(X[0] + 10)

def matyas(X):
    return 0.26 * (X[0]**2 + X[1]**2) - 0.48 * X[0] * X[1]

def levi13(X):
    return np.sin(3 * np.pi * X[0])**2 + ((X[0] - 1)**2 * (1 + np.sin(3 * np.pi * X[1])**2)) + ((X[1] - 1)**2 * (1 + np.sin(2 * np.pi * X[1])**2))

def himmelblau(X):
    return (X[0]**2 + X[1] - 11)**2 + (X[0] + X[1]**2 - 7)**2

def three_hump_camel(X):
    return 2 * X[0]**2 - 1.05 * X[0]**4 + X[0]**6 / 6 + X[0] * X[1] + X[1]**2

def easom(X):
    return -np.cos(X[0]) * np.cos(X[1]) * np.exp(-((X[0] - np.pi)**2 + (X[1] - np.pi)**2))

def cross_in_tray(X):
    return -0.0001 * (np.abs(np.sin(X[0]) * np.sin(X[1]) * np.exp(np.abs(100 - np.sqrt(X[0]**2 + X[1]**2) / np.pi))) + 1)**0.1

def eggholder(X):
    return -(X[1] + 47) * np.sin(np.sqrt(np.abs(X[1] + X[0] / 2 + 47))) - X[0] * np.sin(np.sqrt(np.abs(X[0] - (X[1] + 47))))

def holder_table(X):
    return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0]**2 + X[1]**2) / np.pi)))

def mccormick(X):
    return np.sin(X[0] + X[1]) + (X[0] - X[1])**2 - 1.5*X[0] + 2.5*X[1] + 1

def schaffer2(X):
    return 0.5 + (np.sin(X[0]**2 - X[1]**2)**2 - 0.5) / (1 + 0.001 * (X[0]**2 + X[1]**2))**2

def schaffer4(X):
    return 0.5 + (np.cos(np.sin(np.abs(X[0]**2 - X[1]**2)))**2 - 0.5) / (1 + 0.001 * (X[0]**2 + X[1]**2))**2

def styblinski_tang(X):
    return 0.5 * (X[0]**4 - 16*X[0]**2 + 5*X[0] + X[1]**4 - 16*X[1]**2 + 5*X[1])

def shekel(X, m=10, C=None, beta=None):
    if C is None:
        C = np.array([[4, 4, 4, 4],
                      [1, 1, 1, 1],
                      [8, 8, 8, 8],
                      [6, 6, 6, 6],
                      [3, 7, 3, 7],
                      [2, 9, 2, 9],
                      [5, 5, 3, 3],
                      [8, 1, 8, 1],
                      [6, 2, 6, 2],
                      [7, 3, 7, 3]])
    if beta is None:
        beta = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    
    outer_sum = 0
    for i in range(m):
        inner_sum = 0
        for j in range(2): 
            inner_sum += (X[j] - C[i, j])**2
        outer_sum += 1 / (inner_sum + beta[i])
    return outer_sum
