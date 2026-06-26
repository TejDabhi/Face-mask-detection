import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="😷",
    layout="wide"
)

# --------------------------------
# Title
# --------------------------------
st.title("😷 Face Mask Detection System")
st.write("Detect faces and check whether the person is wearing a mask or not.")

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
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.error("❌ mask_model.keras not found. Put mask_model.keras in the same folder as app.py")
    st.stop()
else:
    st.success("✅ Mask model loaded successfully")

# --------------------------------
# Sidebar Settings
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
    "Min Neighbors",
    min_value=3,
    max_value=10,
    value=5,
    step=1
)

threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

reverse_labels = st.sidebar.checkbox(
    "Reverse Mask / No Mask Labels",
    value=False
)

show_raw_prediction = st.sidebar.checkbox(
    "Show Raw Model Prediction",
    value=True
)

st.sidebar.info(
    "If your result is always opposite, tick Reverse Mask / No Mask Labels."
)

# --------------------------------
# Get Model Input Size
# --------------------------------
def get_input_size(model):
    try:
        input_shape = model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = input_shape[1]
        width = input_shape[2]

        if height is None or width is None:
            return 128, 128

        return int(width), int(height)

    except:
        return 128, 128


# --------------------------------
# Predict Mask Function
# --------------------------------
def predict_mask(face_img):
    input_width, input_height = get_input_size(model)

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Resize same as model training
    face_resized = cv2.resize(face_rgb, (input_width, input_height))

    # Normalize
    face_array = face_resized.astype("float32") / 255.0

    # Add batch dimension
    face_array = np.expand_dims(face_array, axis=0)

    # Predict
    prediction = model.predict(face_array, verbose=0)

    raw_value = float(np.array(prediction).squeeze())

    # Normal logic:
    # raw_value >= threshold means No Mask
    # raw_value < threshold means Mask
    if not reverse_labels:
        if raw_value >= threshold:
            label = "No Mask"
            confidence = raw_value
        else:
            label = "Mask"
            confidence = 1 - raw_value

    # Reverse logic:
    # raw_value >= threshold means Mask
    # raw_value < threshold means No Mask
    else:
        if raw_value >= threshold:
            label = "Mask"
            confidence = raw_value
        else:
            label = "No Mask"
            confidence = 1 - raw_value

    return label, confidence, raw_value


# --------------------------------
# Draw Detection Box
# --------------------------------
def draw_box(image, x, y, w, h, label, confidence):
    if label == "Mask":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    text = f"{label} {confidence * 100:.1f}%"

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

    cv2.rectangle(
        image,
        (x, y - 35),
        (x + 260, y),
        color,
        -1
    )

    cv2.putText(
        image,
        text,
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return image


# --------------------------------
# Image Input
# --------------------------------
image_file = None

if input_method == "Upload Image":
    image_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Take Photo")


# --------------------------------
# Main Logic
# --------------------------------
if image_file is not None:

    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Image could not be read.")
        st.stop()

    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    output_image = image.copy()

    mask_count = 0
    no_mask_count = 0
    raw_predictions = []
    report = []

    for i, (x, y, w, h) in enumerate(faces, start=1):
        face_roi = image[y:y + h, x:x + w]

        label, confidence, raw_value = predict_mask(face_roi)

        raw_predictions.append(raw_value)

        if label == "Mask":
            mask_count += 1
        else:
            no_mask_count += 1

        output_image = draw_box(
            output_image,
            x,
            y,
            w,
            h,
            label,
            confidence
        )

        report.append({
            "Face No.": i,
            "Prediction": label,
            "Confidence": f"{confidence * 100:.2f}%",
            "Raw Model Value": round(raw_value, 4)
        })

    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # --------------------------------
    # Display Images
    # --------------------------------
    st.markdown("---")
    st.subheader("📷 Detection Output")

    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(output_image_rgb, caption="Detected Output", use_container_width=True)

    # --------------------------------
    # Metrics
    # --------------------------------
    st.markdown("---")
    st.subheader("📊 Result Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Faces", len(faces))
    c2.metric("Mask Detected", mask_count)
    c3.metric("No Mask Detected", no_mask_count)

    # --------------------------------
    # Final Conclusion
    # --------------------------------
    st.markdown("---")
    st.subheader("✅ Final Conclusion")

    if len(faces) == 0:
        st.warning("No face detected. Please upload a clear front-facing image.")

    elif mask_count > 0 and no_mask_count == 0:
        st.success("Conclusion: Mask Detected 😷")

    elif no_mask_count > 0 and mask_count == 0:
        st.error("Conclusion: No Mask Detected 🚫")

    else:
        st.warning("Conclusion: Mixed Result. Some faces have mask and some do not.")

    # --------------------------------
    # Raw Prediction Debug
    # --------------------------------
    if show_raw_prediction:
        st.markdown("---")
        st.subheader("🧪 Raw Model Prediction")

        st.write("Raw values:", raw_predictions)

        st.info(
            "If raw value is almost same for every image, like 0.616 again and again, "
            "then your model is not trained properly or dataset labels are wrong."
        )

    # --------------------------------
    # Detailed Report
    # --------------------------------
    if len(report) > 0:
        st.markdown("---")
        st.subheader("📋 Detailed Report")
        st.dataframe(report, use_container_width=True)

else:
    st.info("Upload image or use camera to start detection.")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("Developed using Streamlit, OpenCV and TensorFlow")