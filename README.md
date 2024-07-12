# Optimizador de Funciones

Este repositorio proporciona una colección de algoritmos para la optimización de funciones de una y varias variables. Los métodos incluidos abarcan tanto técnicas directas como basadas en gradientes y derivadas, ofreciendo soluciones para una amplia gama de problemas de optimización. El paquete es fácilmente descargable y utilizable a través de PyPI.

## Métodos Implementados

### Métodos para Funciones de una Variable

#### Métodos de Eliminación de Regiones
- **Método de División de Intervalos por la Mitad**: Divide iterativamente el intervalo a la mitad y selecciona la subregión que contiene el mínimo.
- **Búsqueda de Fibonacci**: Utiliza la secuencia de Fibonacci para minimizar el número de evaluaciones de la función.
- **Método de la Sección Dorada**: Optimiza la búsqueda del mínimo utilizando la razón áurea para reducir el intervalo de búsqueda.

#### Métodos Basados en la Derivada
- **Método de Newton-Raphson**: Utiliza derivadas de la función para encontrar sus raíces, adaptado para encontrar mínimos.
- **Método de Bisección**: Encuentra raíces de funciones univariadas, útil para localizar mínimos en funciones derivadas.
- **Método de la Secante**: Una versión modificada del método de Newton-Raphson que no requiere la segunda derivada de la función.

### Métodos para Funciones Multivariadas

#### Métodos Directos
- **Caminata Aleatoria**: Explora el espacio de búsqueda de manera aleatoria para encontrar un mínimo.
- **Método de Nelder y Mead (Simplex)**: Utiliza un simplejo de puntos para iterativamente aproximarse al mínimo.
- **Método de Hooke-Jeeves**: Un algoritmo de búsqueda directa que combina exploración y patrones de búsqueda.

#### Métodos de Gradiente
- **Método de Cauchy**: Utiliza el gradiente para iterar hacia el mínimo siguiendo la dirección de mayor descenso.
- **Método de Fletcher-Reeves**: Una técnica de gradiente conjugado que mejora la eficiencia del método de Cauchy.
- **Método de Newton**: Utiliza la información de la segunda derivada (Hessiana) para encontrar el mínimo más rápidamente.

## Instalación

Puedes instalar este paquete desde PyPI usando pip:

```bash
pip install optimizador-funciones
```
## Uso
Aquí tienes un ejemplo de cómo usar algunos de los métodos de optimización implementados:

```python
Copiar código
import benchmark_functions as bf
from optimizador_funciones import newton_method, gradiente

# Definir la función de Himmelblau
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Gradiente de la función de Himmelblau
def grad_himmelblau(x):
    return gradiente(himmelblau, x)

# Punto inicial
x0 = [0.0, 0.0]

# Encontrar el mínimo usando el método de Newton
minimo = newton_method(himmelblau, grad_himmelblau, x0)
print(f"Resultado Método de Newton: {minimo}")
```
## Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un pull request o una issue para discutir posibles mejoras o problemas.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

