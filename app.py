# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw

st.title("Разпознаване на ръкописни цифри ✍️🤖")

# Зареждаме предварително обучен модел (можеш да си свалиш готов от Keras)
@st.cache_resource
def load_mnist_model():
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    model = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, verbose=0)  # 1 епоха за бърз тест
    return model

model = load_mnist_model()

# Създаваме бяло поле за рисуване
canvas_size = 280
img = Image.new("L", (canvas_size, canvas_size), 255)
draw = ImageDraw.Draw(img)

st.write("Рисувай цифра (0–9) в полето по-долу:")

# Тук ще използваме Streamlit `st_canvas` (или просто качване на изображение)
user_image = st.file_uploader("Или качи ръкописна цифра (.png)", type=["png","jpg"])

if user_image:
    img = Image.open(user_image).convert("L").resize((28,28))
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,28,28)
    prediction = model.predict(img_array)
    st.write("AI мисли, че това е цифра:", np.argmax(prediction))
