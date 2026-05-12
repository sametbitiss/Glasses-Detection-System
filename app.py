import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Gözlük Tespit Sistemi", page_icon="👓")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('gozluk_tespit_modeli_final.h5')

model = load_my_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("👓 Akıllı Gözlük Tespit Sistemi")
st.write("Bir fotoğraf yükleyin; yapay zeka gözlük olup olmadığını tespit etsin.")

uploaded_file = st.file_uploader("Bir resim seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption='Yüklenen Resim', use_container_width=True)
    
    display_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        st.warning("Resimde yüz tespit edilemedi. Lütfen net bir fotoğraf yükleyin.")
    else:
        for (x, y, w, h) in faces:
            face_crop = img_array[y:y+h, x:x+w]
            
            face_prep = cv2.resize(face_crop, (128, 128))
            face_prep = face_prep.astype('float32') / 255.0
            face_prep = np.expand_dims(face_prep, axis=0)
            
            prediction = model.predict(face_prep)
            prob = prediction[0][0]
            
            if prob > 0.65:
                st.success("### Sonuç: Gözlük Var")
            else:
                st.info("### Sonuç: Gözlük Yok")