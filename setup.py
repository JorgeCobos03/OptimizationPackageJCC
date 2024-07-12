from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = '0.1' 
DESCRIPTION = 'Paquete de Optmización'
LONG_DESCRIPTION = 'Este paquete proporciona una colección de algoritmos para la optimización de funciones de una y varias variables. Los métodos incluidos abarcan tanto técnicas directas como basadas en gradientes y derivadas, ofreciendo soluciones para una amplia gama de problemas de optimización.'

setup(
        name="OptimizationPackageJCC", 
        version=VERSION,
        author="Jorge Cerecedo Cobos",
        author_email="<cobos037r@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['python', 'optimización', 'funciones', 'funciones de una variable', 'funciones multivariadas', 'eliminación de regiones', 'Métodos basados en la derivada', 'Métodos directos', 'Métodos de gradiente'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
