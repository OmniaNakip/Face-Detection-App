import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Face Detection App", layout="wide")
st.title("تطبيق اكتشاف الوجوه (Face Detection) 🕵️‍♂️")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("قم برفع صورة لاكتشاف الوجوه...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة الأصلية", use_column_width=True)
    
    st.write("جاري التحليل...")
    
    results = model.predict(image)
    
    res_plotted = results[0].plot()
    res_image = Image.fromarray(res_plotted[..., ::-1]) 
    
    st.image(res_image, caption="النتيجة بعد التحليل", use_column_width=True)