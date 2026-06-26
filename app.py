import streamlit as st
import cv2
import numpy as np

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Face Detection System",
    page_icon="🙂",
    layout="centered"
)

# --------------------------------
# Title
# --------------------------------
st.title("🙂 Face Detection System")
st.write("Detect faces using OpenCV and Streamlit.")

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
# Sidebar
# --------------------------------
st.sidebar.title("Options")

input_method = st.sidebar.radio(
    "Choose Input Method",
    ["Upload Image", "Use Camera"]
)

scale_factor = st.sidebar.slider(
    "Scale Factor",
    min_value=1.05,
    max_value=1.5,
    value=1.1,
    step=0.05
)

min_neighbors = st.sidebar.slider(
    "Min Neighbors",
    min_value=3,
    max_value=10,
    value=5,
    step=1
)

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
# Face Detection Logic
# --------------------------------
if image_file is not None:

    # Convert image file to OpenCV image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read image.")
        st.stop()

    # Convert BGR to RGB for display
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    # Draw rectangles around faces
    detected_image = image.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(
            detected_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

    # --------------------------------
    # Display Results
    # --------------------------------
    st.subheader("Result")

    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(detected_image, caption="Face Detection Output", use_container_width=True)

    st.success(f"Total Faces Detected: {len(faces)}")

    if len(faces) == 0:
        st.warning("No face detected. Try a clearer image with front-facing face.")

else:
    st.info("Please upload an image or use camera to detect faces.")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("Developed using Streamlit and OpenCV")