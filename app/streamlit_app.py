import json

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


@st.cache_resource
def load():
    model = load_model("models/best_model.h5")
    with open("models/classes.json") as f:
        classes = json.load(f)
    return model, classes


st.title("🧠 BrainScan AI")
st.warning("Decision-support tool only. Not a medical diagnosis.")
model, classes = load()

file = st.file_uploader("Upload a brain MRI", type=["jpg", "jpeg", "png", "bmp"])
if file:
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Cannot read this image.")
        st.stop()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption="Uploaded MRI", width=300)
    x = cv2.resize(img, (224, 224)).astype("float32") / 255.0
    probs = model.predict(np.expand_dims(x, 0), verbose=0)[0]
    st.success(f"Prediction: **{classes[int(np.argmax(probs))]}** ({probs.max():.1%})")
    st.bar_chart(dict(zip(classes, probs.tolist())))
