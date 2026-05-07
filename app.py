import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import time

# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(
    page_title="🎥 Live Object Detection & Tracing",
    page_icon="📸",
    layout="wide"
)

# =========================
# CUSTOM SIDEBAR DESIGN
# =========================
st.markdown("""
<style>
/* Sidebar Background */
[data-testid="stSidebar"] {
    background-color: #6A0DAD;
}

/* Sidebar Text */
[data-testid="stSidebar"] * {
    color: white;
}

/* Developer Card */
.dev-card {
    background-color: rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}

.dev-card h3 {
    margin-bottom: 5px;
    color: #ffffff;
}

.dev-card p {
    margin: 0;
    font-size: 14px;
    color: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SAFE IMAGE PREPROCESSING
# =========================
def preprocess_image(image):
    img = np.array(image)

    # Ensure correct format (fix RGBA issue)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Convert RGB → BGR for YOLO
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# SIDEBAR DASHBOARD
# =========================
with st.sidebar:
    st.title("⚙️ Control Panel")

    app_mode = st.radio(
        "Choose Mode",
        ["📷 Camera Detection", "🖼️ Image Upload"]
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        0.1, 1.0, 0.5, 0.05
    )

    st.markdown("---")
    st.info("YOLOv8 + Streamlit Cloud Ready 🚀")

    # =========================
    # DEVELOPER CARD
    # =========================
    st.markdown("""
    <div class="dev-card">
        <h3>👨‍💻 Developer</h3>
        <p><b>Jenifer P. Magbanua</b></p>
        <p>BSCS - 3A</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# MAIN HEADER
# =========================
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
    🎥 Live Object Detection & Tracing
    </h1>
    <p style='text-align: center;'>
    Point your camera at objects to identify them in real-time.
    </p>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

# =========================
# CAMERA MODE
# =========================
if app_mode == "📷 Camera Detection":
    with col1:
        img_file = st.camera_input("📷 Capture Image")

    if img_file is not None:
        start_time = time.time()

        image = Image.open(img_file).convert("RGB")
        img = preprocess_image(image)

        # SAFE PREDICTION (more stable than model(img))
        results = model.predict(img, conf=conf_threshold)

        annotated = results[0].plot()

        st.image(annotated, caption="Detected Objects", use_container_width=True)

        # =========================
        # OBJECT COUNTING
        # =========================
        counts = {}
        if results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                label = model.names[int(cls)]
                counts[label] = counts.get(label, 0) + 1

        with col2:
            st.subheader("📊 Detection Stats")

            if counts:
                for k, v in counts.items():
                    st.metric(label=k, value=v)
            else:
                st.info("No objects detected")

        st.caption(f"⏱ Processing Time: {round(time.time() - start_time, 2)} sec")

# =========================
# IMAGE UPLOAD MODE
# =========================
elif app_mode == "🖼️ Image Upload":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        start_time = time.time()

        image = Image.open(uploaded_file).convert("RGB")
        img = preprocess_image(image)

        # SAFE PREDICTION
        results = model.predict(img, conf=conf_threshold)

        annotated = results[0].plot()

        st.image(annotated, caption="Detected Objects", use_container_width=True)

        # =========================
        # OBJECT COUNTING
        # =========================
        counts = {}
        if results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                label = model.names[int(cls)]
                counts[label] = counts.get(label, 0) + 1

        st.subheader("📊 Object Summary")
        st.json(counts)

        st.caption(f"⏱ Processing Time: {round(time.time() - start_time, 2)} sec")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center>🚀 YOLOv8 + Streamlit | Stable Deployment Version</center>",
    unsafe_allow_html=True
)