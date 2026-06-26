import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #1f77b4;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}
.result-box {
    background-color: #f0f8ff;
    padding: 25px;
    border-radius: 15px;
    border-left: 6px solid #1f77b4;
    text-align: center;
}
.price-text {
    font-size: 36px;
    font-weight: 800;
    color: #0b8457;
}
.info-box {
    background-color: #f9f9f9;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model and Data
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("pipe.pkl"):
        st.error("❌ pipe.pkl file not found. Please keep pipe.pkl in the same folder as app.py")
        st.stop()

    with open("pipe.pkl", "rb") as file:
        model = pickle.load(file)

    return model


@st.cache_data
def load_data():
    if not os.path.exists("df.pkl"):
        st.error("❌ df.pkl file not found. Please keep df.pkl in the same folder as app.py")
        st.stop()

    with open("df.pkl", "rb") as file:
        data = pickle.load(file)

    return data


pipe = load_model()
df = load_data()

# -----------------------------
# Validate Required Columns
# -----------------------------
required_columns = ["Company", "TypeName", "Cpu brand", "Gpu brand", "os"]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"❌ Missing columns in df.pkl: {missing_columns}")
    st.stop()

# -----------------------------
# Helper Functions
# -----------------------------
def clean_options(column_name):
    return sorted(df[column_name].dropna().unique())


def calculate_ppi(resolution, screen_size):
    x_res = int(resolution.split("x")[0])
    y_res = int(resolution.split("x")[1])
    ppi = ((x_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size
    return ppi


def predict_price(input_query):
    prediction = pipe.predict(input_query)[0]
    final_price = int(np.exp(prediction))
    return final_price


def price_category(price):
    if price < 30000:
        return "Budget Laptop"
    elif price < 70000:
        return "Mid-Range Laptop"
    elif price < 120000:
        return "Premium Laptop"
    else:
        return "High-End / Gaming / Workstation Laptop"


# -----------------------------
# Session State
# -----------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -----------------------------
# Header
# -----------------------------
st.markdown("<div class='main-title'>💻 Laptop Price Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Predict laptop price based on brand, RAM, SSD, CPU, GPU, display and other features</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📌 About Project")
    st.write("This app predicts laptop price using a trained Machine Learning model.")

    st.markdown("### Files Required")
    st.write("✅ app.py")
    st.write("✅ pipe.pkl")
    st.write("✅ df.pkl")
    st.write("✅ requirements.txt")

    st.markdown("### Input Tips")
    st.write("Select proper laptop configuration for better prediction accuracy.")

    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        st.success("History cleared successfully.")

# -----------------------------
# Main Input Form
# -----------------------------
st.markdown("## Enter Laptop Details")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        company = st.selectbox("Brand", clean_options("Company"))
        laptop_type = st.selectbox("Laptop Type", clean_options("TypeName"))
        ram = st.selectbox("RAM in GB", [2, 4, 6, 8, 12, 16, 24, 32, 64])
        weight = st.number_input(
            "Weight of Laptop in KG",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1
        )

    with col2:
        touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
        ips = st.selectbox("IPS Display", ["No", "Yes"])
        screen_size = st.slider(
            "Screen Size in Inches",
            min_value=10.0,
            max_value=18.0,
            value=15.6,
            step=0.1
        )
        resolution = st.selectbox(
            "Screen Resolution",
            [
                "1920x1080",
                "1366x768",
                "1600x900",
                "3840x2160",
                "3200x1800",
                "2880x1800",
                "2560x1600",
                "2560x1440",
                "2304x1440"
            ]
        )

    with col3:
        cpu = st.selectbox("CPU Brand", clean_options("Cpu brand"))
        hdd = st.selectbox("HDD in GB", [0, 128, 256, 512, 1024, 2048])
        ssd = st.selectbox("SSD in GB", [0, 8, 128, 256, 512, 1024])
        gpu = st.selectbox("GPU Brand", clean_options("Gpu brand"))
        os_name = st.selectbox("Operating System", clean_options("os"))

    submitted = st.form_submit_button("🔮 Predict Price")

# -----------------------------
# Prediction Logic
# -----------------------------
if submitted:

    touchscreen_value = 1 if touchscreen == "Yes" else 0
    ips_value = 1 if ips == "Yes" else 0

    ppi = calculate_ppi(resolution, screen_size)

    query = np.array([
        company,
        laptop_type,
        ram,
        weight,
        touchscreen_value,
        ips_value,
        ppi,
        cpu,
        hdd,
        ssd,
        gpu,
        os_name
    ])

    query = query.reshape(1, 12)

    try:
        predicted_price = predict_price(query)
        category = price_category(predicted_price)

        st.markdown("---")

        col_result1, col_result2 = st.columns([2, 1])

        with col_result1:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown("### Predicted Laptop Price")
            st.markdown(
                f"<div class='price-text'>₹ {predicted_price:,}</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"### Category: {category}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_result2:
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("### Display Info")
            st.write(f"**Resolution:** {resolution}")
            st.write(f"**Screen Size:** {screen_size} inches")
            st.write(f"**Calculated PPI:** {round(ppi, 2)}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Save History
        st.session_state.prediction_history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Company": company,
            "Type": laptop_type,
            "RAM": ram,
            "Weight": weight,
            "Touchscreen": touchscreen,
            "IPS": ips,
            "Screen Size": screen_size,
            "Resolution": resolution,
            "CPU": cpu,
            "HDD": hdd,
            "SSD": ssd,
            "GPU": gpu,
            "OS": os_name,
            "Predicted Price": predicted_price,
            "Category": category
        })

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write("Error details:")
        st.code(str(e))

# -----------------------------
# Configuration Summary
# -----------------------------
st.markdown("---")
st.markdown("## Selected Configuration Preview")

preview_data = {
    "Feature": [
        "Brand",
        "Type",
        "RAM",
        "Weight",
        "Touchscreen",
        "IPS",
        "Screen Size",
        "Resolution",
        "CPU",
        "HDD",
        "SSD",
        "GPU",
        "OS"
    ],
    "Value": [
        company,
        laptop_type,
        f"{ram} GB",
        f"{weight} KG",
        touchscreen,
        ips,
        f"{screen_size} inches",
        resolution,
        cpu,
        f"{hdd} GB",
        f"{ssd} GB",
        gpu,
        os_name
    ]
}

preview_df = pd.DataFrame(preview_data)
st.dataframe(preview_df, use_container_width=True, hide_index=True)

# -----------------------------
# Prediction History
# -----------------------------
if len(st.session_state.prediction_history) > 0:
    st.markdown("---")
    st.markdown("## Prediction History")

    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History as CSV",
        data=csv,
        file_name="laptop_prediction_history.csv",
        mime="text/csv"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit and Machine Learning")