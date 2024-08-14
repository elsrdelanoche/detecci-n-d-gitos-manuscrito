import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk

# Función para mostrar gráficos en tkinter
def plot_in_tkinter():
    # Crear la figura
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfico de precisión
    axs[0].plot(history.history['accuracy'], label='Precisión en el entrenamiento')
    axs[0].plot(history.history['val_accuracy'], label='Precisión en la validación')
    axs[0].set_xlabel('Épocas')
    axs[0].set_ylabel('Precisión')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Precisión durante el entrenamiento y validación')

    # Gráfico de pérdida
    axs[1].plot(history.history['loss'], label='Pérdida en el entrenamiento')
    axs[1].plot(history.history['val_loss'], label='Pérdida en la validación')
    axs[1].set_xlabel('Épocas')
    axs[1].set_ylabel('Pérdida')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Pérdida durante el entrenamiento y validación')

    # Crear la ventana principal de tkinter
    root = tk.Tk()
    root.wm_title("Resultados del Entrenamiento")

    # Agregar la figura a la ventana tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Botón para cerrar la ventana
    quit_button = tk.Button(root, text="Cerrar", command=root.quit)
    quit_button.pack(side=tk.BOTTOM)

    # Ejecutar el loop de tkinter
    tk.mainloop()

# Cargar el dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar las imágenes a valores entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Crear el modelo de red neuronal
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

model.save('mnist_model.h5')  # Guarda el modelo entrenado

# Mostrar los gráficos en tkinter
plot_in_tkinter()

