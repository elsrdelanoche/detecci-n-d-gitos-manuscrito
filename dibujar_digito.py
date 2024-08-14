import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('mnist_model.h5')

# Función para predecir el dígito dibujado
def predict_digit(image):
    # Preprocesar la imagen
    image = np.array(image).reshape(1, 28, 28) / 255.0
    pred = model.predict(image)
    return np.argmax(pred), np.max(pred)

# Función para solicitar retroalimentación del usuario
def get_user_feedback(predicted_digit):
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    is_correct = simpledialog.askstring("Feedback", f"El modelo predijo: {predicted_digit}. ¿Es correcto? (sí/no)")
    root.destroy()
    return is_correct.lower() == 'sí'

# Dibujar en un canvas para que el usuario haga un dígito
def draw_digit():
    root = tk.Tk()
    root.title("Dibuja un dígito")

    canvas = tk.Canvas(root, width=280, height=280, bg='white')
    canvas.pack()

    image = np.zeros((28, 28), dtype=np.uint8)

    def paint(event):
        x, y = event.x // 10, event.y // 10
        canvas.create_rectangle(x * 10, y * 10, (x + 1) * 10, (y + 1) * 10, fill='black')
        image[y, x] = 255

    canvas.bind("<B1-Motion>", paint)

    def save_and_predict():
        root.destroy()
        plt.imshow(image, cmap='gray')
        plt.show()
        digit, confidence = predict_digit(image)
        print(f"Predicción: {digit} con una confianza del {confidence:.3f}")
        if get_user_feedback(digit):
            # Si el usuario confirma que la predicción es correcta, se reentrena el modelo
            global x_train, y_train
            x_train = np.append(x_train, [image], axis=0)
            y_train = np.append(y_train, [digit], axis=0)
            
            # Recrear el optimizador
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Reentrenar el modelo
            model.fit(x_train, y_train, epochs=1, verbose=1)
            model.save('mnist_model.keras')
        else:
            print("Predicción incorrecta. No se guardarán los datos.")

    button = tk.Button(root, text="Predecir", command=save_and_predict)
    button.pack()

    root.mainloop()

# Cargar los datos MNIST originales para el reentrenamiento
(x_train, y_train), _ = datasets.mnist.load_data()
x_train = x_train / 255.0

# Llamar a la función para dibujar y predecir un dígito
draw_digit()
