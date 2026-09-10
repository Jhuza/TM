import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# Configuración de la página
st.set_page_config(
    page_title="Hábitat Inteligente",
    page_icon="🏠",
    layout="centered"
)

# Estilos visuales
st.markdown("""
<style>
    .welcome-card {
        padding: 28px;
        margin-top: 20px;
        border-radius: 20px;
        text-align: center;
        color: white;
        background: linear-gradient(135deg, #1761C9, #0F393B);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
    }

    .welcome-card h1 {
        margin-bottom: 10px;
        font-size: 34px;
    }

    .welcome-card h2 {
        margin: 0;
        font-size: 21px;
        font-weight: 500;
        line-height: 1.4;
    }

    .probability {
        margin-top: 16px;
        font-size: 15px;
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

# Carga del modelo
model = load_model("keras_model.h5", compile=False)

st.title("🏠 Hábitat inteligente")
st.write("Sistema de acceso mediante reconocimiento de imágenes")
st.caption(f"Versión de Python: {platform.python_version()}")

# Imagen inicial
image = Image.open("OIG5.jpg")
st.image(image, width=350)

with st.sidebar:
    st.subheader("¿Cómo funciona?")
    st.write(
        "El sistema usa un modelo entrenado en Teachable Machine "
        "para identificar a Juanjo y reconocer su gesto de acceso."
    )

# Captura desde la cámara
img_file_buffer = st.camera_input("📷 Realiza tu gesto frente a la cámara")

if img_file_buffer is not None:

    # Abrir y preparar la imagen
    img = Image.open(img_file_buffer).convert("RGB")
    img = img.resize((224, 224))

    # Convertirla en un arreglo de NumPy
    img_array = np.asarray(img)

    # Normalizar la imagen
    normalized_image_array = (
        img_array.astype(np.float32) / 127.5
    ) - 1

    # Crear el arreglo de entrada
    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )
    data[0] = normalized_image_array

    # Ejecutar la predicción
    prediction = model.predict(data, verbose=0)

    probabilidad_cara = float(prediction[0][0])
    probabilidad_gesto = float(prediction[0][1])

    # Mostrar resultados
    st.subheader("Resultado de la detección")
    st.write(f"👤 Reconocimiento normal: **{probabilidad_cara:.1%}**")
    st.write(f"👋 Gesto de acceso: **{probabilidad_gesto:.1%}**")
    st.progress(probabilidad_gesto)

    # Activar el hogar al reconocer el gesto
    if probabilidad_gesto > 0.5:
        st.balloons()

        st.markdown(f"""
        <div class="welcome-card">
            <h1>🏠✨ <strong>¡Bienvenido a tu hábitat, Juanjo!</strong></h1>
            <h2>
                Ya puedes descansar en tu hogar después del ajetreo del día 😌🛋️
            </h2>
            <div class="probability">
                Gesto reconocido con una confianza del
                <strong>{probabilidad_gesto:.1%}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info(
            "👋 No se reconoció el gesto de acceso. "
            "Intenta tomar otra fotografía con mejor iluminación."
        )
