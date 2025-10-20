import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")
st.title("🖼️ Simple Image Classifier (Streamlit + FastAPI)")

backend_url = st.text_input("Backend URL", value="http://host.docker.internal:8000/predict")


uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Preview", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            # send to FastAPI
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            try:
                resp = requests.post(backend_url, files=files, timeout=30)
                if resp.ok:
                    data = resp.json()
                    st.success(f"Label: **{data.get('label','(no label)')}**")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
