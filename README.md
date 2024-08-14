# Proyecto de Reconocimiento de Dígitos Manuscritos

Este proyecto utiliza una red neuronal para reconocer dígitos manuscritos basándose en el famoso dataset MNIST. El proyecto incluye dos scripts principales: uno para entrenar el modelo y otro para predecir dígitos a partir de entradas dibujadas por el usuario.

## Requisitos

- Python 3.10+
- TensorFlow
- NumPy
- Matplotlib
- Tkinter (para la interfaz de dibujo)

### Instalación de Dependencias

```bash
pip install tensorflow numpy matplotlib
```

### Entrenamiento del Modelo

El script `entrenar_modelo.py` entrena una red neuronal simple utilizando el dataset MNIST y guarda el modelo entrenado en un archivo. 

#### Uso:

```bash
python entrenar_modelo.py
```

El modelo se guardará como `mnist_model.h5`.

### Predicción de Dígitos y Reentrenamiento

El script `dibujar_digito.py` permite al usuario dibujar un dígito en una ventana emergente, predice el dígito y le permite al usuario confirmar si la predicción es correcta. Si es correcta, el modelo se reentrena con el nuevo dígito y se guarda.

#### Uso:

```bash
python dibujar_digito.py
```

### Funcionalidades Clave:

- **Predicción en Tiempo Real:** Predice el dígito dibujado por el usuario en una interfaz gráfica.
- **Reentrenamiento Incremental:** Si el usuario confirma que la predicción es correcta, el modelo se reentrena brevemente con el nuevo dato.
- **Guardado del Modelo:** El modelo se guarda en el formato nativo de Keras (`.keras`) tras cada reentrenamiento.

### Estructura del Proyecto:

- `entrenar_modelo.py`: Script para entrenar el modelo inicial.
- `dibujar_digito.py`: Script para predecir y reentrenar el modelo basado en la entrada del usuario.
- `mnist_model.h5`: Archivo que contiene el modelo entrenado.

### Notas:

- Si se encuentra un error de memoria, considera aumentar la capacidad de tu sistema o ajustar la frecuencia de reentrenamiento.
