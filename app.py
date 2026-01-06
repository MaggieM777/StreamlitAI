import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Разпознаване на ръкописни цифри", page_icon="✍️")
st.title("Разпознаване на ръкописни цифри ✍️🤖")
st.write("Качи ръкописна цифра и AI ще се опита да я разпознае.")

# -----------------------------
# Трениране на модел
# -----------------------------
@st.cache_resource
def train_model():
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1)) / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),  # повече слоеве за по-добра точност
        max_iter=500,  # повече итерации
        random_state=42,
        verbose=False,
        early_stopping=True
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    st.sidebar.success(f"Моделът е обучен с точност: {accuracy:.2%}")
    return model

model = train_model()

# -----------------------------
# Качване на изображение
# -----------------------------
st.sidebar.header("Настройки")
st.sidebar.write("""
Моделът е обучен с MNIST 8x8.
За по-добри резултати:
- Черна цифра на бял фон
- Минимум шум
- Центрирана цифра
""")

user_image = st.file_uploader("Качи ръкописна цифра (.png/.jpg/.jpeg)", type=["png", "jpg", "jpeg"])

if user_image:
    try:
        # Отваряне и обработка на изображението
        img = Image.open(user_image).convert("L")  # преобразуване в grayscale
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Оригинално изображение")
            st.image(img, caption=f"Размер: {img.size}", use_column_width=True)
        
        # Преоразмеряване на 8x8 (както в MNIST)
        img_resized = img.resize((8, 8), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        
        # Инверсия ако фонът е тъмен (черна цифра на бял фон е стандартно)
        if img_array.mean() > 128:
            img_array = 255 - img_array
        
        # Нормализиране както при обучението (0-16)
        img_array = img_array / 16.0
        img_flat = img_array.reshape(1, 64)
        
        # Прогноза
        prediction = model.predict(img_flat)[0]
        probabilities = model.predict_proba(img_flat)[0]
        
        with col2:
            st.subheader("Обработено за модела (8x8)")
            st.image(img_resized.resize((64, 64)), caption="8x8 увеличено", use_column_width=False)
        
        # Резултати
        st.markdown("---")
        st.subheader("📊 Резултат")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown(f"### 🎯 Прогноза: **{prediction}**")
            st.markdown(f"**Увереност:** {probabilities[prediction]:.2%}")
        
        with col_res2:
            st.markdown("### Вероятности за всички цифри:")
            prob_dict = {i: prob for i, prob in enumerate(probabilities)}
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            for digit, prob in sorted_probs[:3]:  # Топ 3 прогнози
                st.progress(float(prob), text=f"Цифра {digit}: {prob:.2%}")
        
        # Показване на всички вероятности
        with st.expander("Виж всички вероятности"):
            for digit in range(10):
                st.write(f"Цифра {digit}: {probabilities[digit]:.4f}")
                
    except Exception as e:
        st.error(f"Грешка при обработка на изображението: {str(e)}")
        st.info("Моля, опитайте с друго изображение.")

# -----------------------------
# Инструкции
# -----------------------------
st.markdown("---")
st.markdown("""
### 📝 Инструкции:
1. Качете изображение с ръкописна цифра (0-9)
2. Изображението ще бъде автоматично обработено
3. AI моделът ще даде прогноза
4. За по-добри резултати използвайте ясни изображения

### ℹ️ Забележки:
- Моделът е обучен с 8x8 пиксела изображения
- Точността е около 95-97%
- За най-добри резултати цифрата трябва да е центрирана
""")
