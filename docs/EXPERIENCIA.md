# Hábitat inteligente

Usa un modelo entrenado para distinguir rostro y gesto en una captura.

- Entrada: Fotografía de la cámara.
- Resultado: Probabilidades de dos clases y mensaje de acceso según el umbral.
- Tecnología: Teachable Machine · Keras · TensorFlow

## Uso

Toma una foto y observa las probabilidades de rostro y gesto. El mensaje cambia cuando la probabilidad de gesto supera 0,5.

## Límites

Es una demostración de clasificación de dos clases, no un sistema de autenticación seguro.

## Evidencia de ejecución

La ficha describe el funcionamiento del código. Una ejecución del modelo o una conexión a un dispositivo debe documentarse con su entrada, resultado observado y captura real; no se presentan resultados simulados como evidencia académica.

[Demo](https://csxqwm99qyob4zptxventz.streamlit.app/) · [Código](https://github.com/Jhuza/TM)
