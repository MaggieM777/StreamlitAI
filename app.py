import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

st.title("Разпознаване на ръкописни цифри ✍️🤖")
st.write("Качи ръкописна цифра или я нарисувай и AI ще се опита да я разпознае.")

# -----------------------------
# Зареждаме dataset и тренираме модел
# -----------------------------
@st.cache_resource
def train_model():
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1)) / 16.0  # нормализиране
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# -----------------------------
# Качване на изображение
# -----------------------------
user_image = st.file_uploader("Качи ръкописна цифра (.png/.jpg)", type=["png","jpg"])

if user_image:
    img = Image.open(user_image).convert("L").resize((8,8))  # sklearn MNIST е 8x8
    img_array = np.array(img)
    
    # инверсия (ако е бяло поле и черна цифра)
    if img_array.mean() > 128:
        img_array = 255 - img_array
    
    img_array = img_array / 16.0  # нормализиране
    img_array = img_array.reshape(1, 64)
    
    prediction = model.predict(img_array)
    st.image(img.resize((64,64)), caption="Въведенa цифра", use_column_width=False)
    st.write("AI мисли, че това е цифра:", prediction[0])
