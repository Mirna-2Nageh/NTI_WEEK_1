import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
from streamlit_drawable_canvas import st_canvas

# ---------------- Load Trained Model ----------------
@st.cache_resource
def load_trained_model():
    return load_model("mnist_cnn_enhanced.h5")

model = load_trained_model()

# ---------------- Preprocessing Function ----------------
def preprocess_image(img_array):
    try:
        # Convert to grayscale
        if img_array.shape[-1] == 4:  # RGBA (from canvas)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Auto-invert if background is white
        if np.mean(img) > 127:
            img = cv2.bitwise_not(img)

        # Threshold (binary)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Skip if nothing drawn
        if np.sum(img) == 0:
            return None

        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        # Get bounding box of largest contour
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = img[y:y+h, x:x+w]

        # Resize to 20x20
        digit = cv2.resize(digit, (20, 20))

        # Pad to 28x28
        padded = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - 20) // 2
        y_offset = (28 - 20) // 2
        padded[y_offset:y_offset+20, x_offset:x_offset+20] = digit

        # Normalize
        padded = padded.astype("float32") / 255.0
        padded = padded.reshape(1, 28, 28, 1)

        return padded
    except Exception:
        return None

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="MNIST Digit Recognition", page_icon="🖊️", layout="centered")

st.title("🖊️ MNIST Digit Recognition")
st.write("Upload or draw a handwritten digit (0–9) and let the trained CNN predict it!")

# Sidebar Instructions
st.sidebar.header("ℹ️ How to use")
st.sidebar.markdown("""
1. Upload a **digit image (PNG/JPG)** OR draw a digit.  
2. Ensure the digit is clear, dark on light background.  
3. View preprocessing and predictions.  
""")

# ---- Upload Section ----
uploaded_file = st.file_uploader("📂 Upload a digit image", type=["png", "jpg", "jpeg"])

# ---- Drawing Section ----
st.subheader("✏️ Or draw a digit below")
canvas = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Use uploaded image if provided, otherwise check canvas
img = None
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif canvas.image_data is not None:
    img = canvas.image_data.astype("uint8")

if img is not None:
    st.image(img, caption="🖼️ Input Image", use_container_width=True)

    processed = preprocess_image(img)

    if processed is not None:
        st.image(processed.reshape(28, 28), caption="✨ Preprocessed (MNIST Style)", width=200, channels="GRAY")

        # Predict
        prediction = model.predict(processed, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"✅ Predicted Digit: **{digit}** with {confidence*100:.2f}% confidence")

        # Plot probabilities
        st.bar_chart(pd.DataFrame(prediction[0], index=range(10), columns=["Probability"]))
    else:
        st.error("⚠️ No digit detected in the image. Try again.")
