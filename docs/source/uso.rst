USO
=====

.. _Instalación:

Instalación
------------

Este paquete puede ser instalado fácilmente utilizando `pip`, el gestor de paquetes de Python. A continuación se detallan los pasos necesarios para instalar `OptimizationPackageJCC` desde el índice de prueba de PyPI. Si ta lo tienes instalado peudes salatar directamente a la seccion :ref:`Métodos para Funciones de una Variable`.

Primero, asegúrate de tener `pip` instalado en tu sistema. Si no lo tienes, puedes instalarlo siguiendo las instrucciones en la documentación oficial de [pip](https://pip.pypa.io/en/stable/installation/).

Para instalar el paquete `OptimizationPackageJCC`, ejecuta el siguiente comando en tu terminal:

.. code-block:: bash

    pip install -i https://test.pypi.org/simple/ OptimizationPackageJCC==0.0.1

Este comando instalará la versión 0.0.1 del paquete desde el índice de prueba de PyPI.

**Nota**: El índice de prueba de PyPI (`test.pypi.org`) es una versión separada de PyPI que se utiliza para probar la distribución de paquetes. No se recomienda utilizar este índice para instalaciones en entornos de producción. Para versiones de producción, asegúrate de instalar desde el índice principal de PyPI (`pypi.org`).

Si prefieres instalar desde el índice principal de PyPI, usa el siguiente comando (asegúrate de que el paquete esté publicado en PyPI):

.. code-block:: bash

    pip install optimizationpackagejcc

**Requisitos Previos**

Asegúrate de tener las siguientes dependencias instaladas antes de proceder con la instalación del paquete:
- Python 3.6 o superior
- pip

Para verificar la versión de Python y pip instaladas en tu sistema, puedes usar los siguientes comandos:

.. code-block:: bash

    python --version
    pip --version

**Solución de Problemas**

Si encuentras problemas durante la instalación, aquí tienes algunos pasos que puedes seguir para resolverlos:

1. **Actualizar pip**: Asegúrate de que `pip` esté actualizado a la última versión.
   
   .. code-block:: bash

       pip install --upgrade pip

2. **Entornos Virtuales**: Se recomienda utilizar un entorno virtual para gestionar las dependencias del proyecto. Puedes crear y activar un entorno virtual usando `venv`:

   .. code-block:: bash

       python -m venv myenv
       source myenv/bin/activate  # En Windows usa `myenv\Scripts\activate`

3. **Dependencias Faltantes**: Si faltan dependencias, `pip` debería manejarlas automáticamente. Si no, instala las dependencias manualmente listadas en `requirements.txt` o según la documentación del paquete.

Si después de seguir estos pasos aún tienes problemas, por favor abre un issue en el repositorio del proyecto en GitHub para obtener asistencia.
