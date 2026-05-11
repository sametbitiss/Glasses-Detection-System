import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# 1. Sayfa Ayarları ve Modeli Yükle
st.set_page_config(page_title="Gözlük Tespit Sistemi", page_icon="👓")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('gozluk_tespit_modeli_final.h5')

model = load_my_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- ARAYÜZ ---
st.title("👓 Akıllı Gözlük Tespit Sistemi")
st.write("Bir fotoğraf yükleyin veya kameranızla bir fotoğraf çekin; yapay zeka gözlük olup olmadığını tespit etsin.")

# İki seçenek sunalım: Dosya Yükle veya Fotoğraf Çek
option = st.radio("Yöntem Seçin:", ("Fotoğraf Yükle", "Kamerayı Kullan"))

image = None

if option == "Fotoğraf Yükle":
    uploaded_file = st.file_uploader("Bir resim seçin...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    cam_file = st.camera_input("Fotoğraf Çek")
    if cam_file is not None:
        image = Image.open(cam_file)

if image is not None:
    img_array = np.array(image)
    # Streamlit RGB verir, OpenCV için BGR'a çevirmemiz gerekebilir ama 
    # model RGB eğitildiyse RGB göndermeliyiz. Karmaşayı önleyelim:
    display_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) # OpenCV çizimleri için BGR yapalım
    
    gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # 1. Yüzü Kes (Orijinal RGB resimden kesiyoruz!)
        face_crop = img_array[y:y+h, x:x+w]
        
        # 2. Ön İşleme (Görüntüyü biraz yumuşatalım ki kumlanma gitsin)
        face_crop = cv2.GaussianBlur(face_crop, (3, 3), 0) # Hafif blur kumlanmayı temizler
        
        face_prep = cv2.resize(face_crop, (128, 128))
        face_prep = face_prep.astype('float32') / 255.0
        face_prep = np.expand_dims(face_prep, axis=0)
        
        # 3. Tahmin
        prediction = model.predict(face_prep)
        prob = prediction[0][0]
        
        # Eşik Değerini (Threshold) Biraz Yükseltelim
        # Model çok hassas olabilir, 0.5 yerine 0.6 veya 0.7 deneyebilirsin
        if prob > 0.65: # %65 ve üzeri kesinlikte gözlük var desin
            res_text = f"GOZLUKLU (%{prob*100:.1f})"
            color = (0, 255, 0) # Yeşil (BGR'da Yeşil yine ortadadır)
        else:
            res_text = f"GOZLUKSUZ (%{(1-prob)*100:.1f})"
            color = (0, 0, 255) # Kırmızı (BGR formatında 0,0,255 kırmızıdır)

        cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 4)
        cv2.putText(display_img, res_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Streamlit'e geri gönderirken tekrar RGB yapalım
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    st.image(display_img, use_column_width=True)

st.divider()
st.caption("Görüntü İşleme Ödevi - 2026 | Model Başarısı: %98.40")