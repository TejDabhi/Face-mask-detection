import streamlit as st
import cv2
import numpy as np
import os
import tensorflow as tf

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="😷",
    layout="wide"
)

# --------------------------------
# Custom CSS
# --------------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f5f5;
    border: 1px solid #ddd;
}
.success-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #e8f8f0;
    border-left: 6px solid #16a34a;
}
.danger-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #fdecec;
    border-left: 6px solid #dc2626;
}
.warning-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #fff7ed;
    border-left: 6px solid #f97316;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Header
# --------------------------------
st.markdown("<div class='main-title'>😷 Face Mask Detection System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Detect faces and classify whether a person is wearing a mask or not</div>",
    unsafe_allow_html=True
)

# --------------------------------
# Load Face Detector
# --------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("Face detector could not be loaded.")
    st.stop()

# --------------------------------
# Load Mask Model
# --------------------------------
MODEL_PATH = "mask_model.keras"

@st.cache_resource
def load_mask_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

mask_model = load_mask_model()

# --------------------------------
# Helper Functions
# --------------------------------
def get_model_input_size(model):
    try:
        input_shape = model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = input_shape[1]
        width = input_shape[2]

        if height is None or width is None:
            return 128, 128

        return int(width), int(height)

    except Exception:
        return 128, 128


def predict_mask(face_img, model, threshold, positive_label, class_0_label, class_1_label):
    input_width, input_height = get_model_input_size(model)

    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (input_width, input_height))
    face_array = face_resized.astype("float32") / 255.0
    face_array = np.expand_dims(face_array, axis=0)

    prediction = model.predict(face_array, verbose=0)
    prediction = np.array(prediction).squeeze()

    # Case 1: Binary sigmoid model
    if prediction.ndim == 0 or prediction.size == 1:
        score = float(prediction)

        if positive_label == "Mask":
            if score >= threshold:
                label = "Mask"
                confidence = score
            else:
                label = "No Mask"
                confidence = 1 - score
        else:
            if score >= threshold:
                label = "No Mask"
                confidence = score
            else:
                label = "Mask"
                confidence = 1 - score

        return label, confidence

    # Case 2: Softmax model with 2 classes
    else:
        probabilities = prediction.flatten()
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        if class_id == 0:
            label = class_0_label
        elif class_id == 1:
            label = class_1_label
        else:
            label = f"Class {class_id}"

        return label, confidence


def draw_label_box(image, x, y, w, h, label, confidence):
    if label == "Mask":
        color = (0, 255, 0)
    elif label == "No Mask":
        color = (0, 0, 255)
    else:
        color = (255, 165, 0)

    text = f"{label} ({confidence * 100:.1f}%)"

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

    y_text = y - 10 if y - 10 > 10 else y + 25

    cv2.rectangle(
        image,
        (x, y_text - 25),
        (x + len(text) * 12, y_text + 5),
        color,
        -1
    )

    cv2.putText(
        image,
        text,
        (x + 5, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return image


# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.header("⚙️ Settings")

input_method = st.sidebar.radio(
    "Choose Input Method",
    ["Upload Image", "Use Camera"]
)

scale_factor = st.sidebar.slider(
    "Face Detection Scale Factor",
    min_value=1.05,
    max_value=1.50,
    value=1.10,
    step=0.05
)

min_neighbors = st.sidebar.slider(
    "Face Detection Min Neighbors",
    min_value=3,
    max_value=10,
    value=5,
    step=1
)

threshold = st.sidebar.slider(
    "Mask Prediction Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Label Mapping")

positive_label = st.sidebar.selectbox(
    "For binary model, prediction >= threshold means:",
    ["Mask", "No Mask"]
)

class_0_label = st.sidebar.selectbox(
    "For 2-class model, Class 0 means:",
    ["No Mask", "Mask"],
    index=0
)

class_1_label = st.sidebar.selectbox(
    "For 2-class model, Class 1 means:",
    ["Mask", "No Mask"],
    index=0
)

st.sidebar.info("If result is opposite, change the label mapping above.")

# --------------------------------
# Model Status
# --------------------------------
if mask_model is None:
    st.warning(
        "mask_model.keras not found. App will detect faces only. "
        "For Mask / No Mask result, keep mask_model.keras in the same folder as app.py."
    )
else:
    st.success("Mask model loaded successfully.")

# --------------------------------
# Image Input
# --------------------------------
image_file = None

if input_method == "Upload Image":
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Take a photo")

# --------------------------------
# Main Detection Logic
# --------------------------------
if image_file is not None:

    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read image.")
        st.stop()

    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    output_image = image.copy()

    mask_count = 0
    no_mask_count = 0
    unknown_count = 0

    report_data = []

    for index, (x, y, w, h) in enumerate(faces, start=1):
        face_roi = image[y:y + h, x:x + w]

        if mask_model is not None:
            label, confidence = predict_mask(
                face_roi,
                mask_model,
                threshold,
                positive_label,
                class_0_label,
                class_1_label
            )

            if label == "Mask":
                mask_count += 1
            elif label == "No Mask":
                no_mask_count += 1
            else:
                unknown_count += 1

        else:
            label = "Face Detected"
            confidence = 1.0
            unknown_count += 1

        output_image = draw_label_box(
            output_image,
            x,
            y,
            w,
            h,
            label,
            confidence
        )

        report_data.append({
            "Face No.": index,
            "Result": label,
            "Confidence": f"{confidence * 100:.2f}%"
        })

    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # --------------------------------
    # Display Images
    # --------------------------------
    st.markdown("---")
    st.subheader("📷 Detection Result")

    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(output_image_rgb, caption="Output Image", use_container_width=True)

    # --------------------------------
    # Summary
    # --------------------------------
    st.markdown("---")
    st.subheader("📊 Analysis Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Faces", len(faces))
    c2.metric("Mask Detected", mask_count)
    c3.metric("No Mask Detected", no_mask_count)
    c4.metric("Unknown", unknown_count)

    # --------------------------------
    # Final Conclusion
    # --------------------------------
    st.markdown("---")
    st.subheader("✅ Final Conclusion")

    if len(faces) == 0:
        st.markdown(
            "<div class='warning-box'><h3>No Face Detected</h3>"
            "<p>Please upload a clear front-facing face image.</p></div>",
            unsafe_allow_html=True
        )

    elif mask_model is None:
        st.markdown(
            "<div class='warning-box'><h3>Face Detected, But Mask Model Missing</h3>"
            "<p>The app detected face/faces, but cannot say Mask or No Mask because mask_model.keras is missing.</p></div>",
            unsafe_allow_html=True
        )

    elif no_mask_count == 0 and mask_count > 0:
        st.markdown(
            "<div class='success-box'><h3>All Detected Faces Are Wearing Mask 😷</h3>"
            "<p>Conclusion: Mask Detected.</p></div>",
            unsafe_allow_html=True
        )

    elif no_mask_count > 0 and mask_count == 0:
        st.markdown(
            "<div class='danger-box'><h3>No Mask Detected 🚫</h3>"
            "<p>Conclusion: Person is not wearing a mask.</p></div>",
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            "<div class='warning-box'><h3>Mixed Result</h3>"
            "<p>Some people are wearing masks and some are not wearing masks.</p></div>",
            unsafe_allow_html=True
        )

    # --------------------------------
    # Detailed Report
    # --------------------------------
    if len(report_data) > 0:
        st.markdown("---")
        st.subheader("📋 Detailed Face Report")
        st.dataframe(report_data, use_container_width=True)

else:
    st.info("Please upload an image or use camera to start detection.")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("Developed using Streamlit, OpenCV and TensorFlow")