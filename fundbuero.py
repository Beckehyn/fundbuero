import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# 1. Seite konfigurieren (Optional, sieht aber schöner aus)
st.set_page_config(page_title="KI Fundbüro", layout="centered")
st.title("🔍 KI Fundbüro Erkennung")

# 2. Modell laden (mit Cache, damit es nur 1x geladen wird)
@st.cache_resource
def load_my_model():
    return load_model("keras_model.h5", compile=False)

model = load_my_model()
class_names = open("labels.txt", "r").readlines()

# 3. Interaktiver Upload
uploaded_file = st.file_uploader("Lade ein Foto des Fundstücks hoch...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild vorbereiten
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Dein Foto', use_column_width=True)
    
    # Vorverarbeitung (Preprocessing)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage treffen
    with st.spinner('KI analysiert das Bild...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip() # .strip() entfernt unschöne Zeilenumbrüche
        confidence_score = prediction[0][index]

    # 4. Ergebnis auf der Webseite ausgeben
    st.divider()
    st.subheader(f"Ergebnis: {class_name[2:]}")
    st.progress(float(confidence_score))
    st.write(f"Sicherheit: **{confidence_score:.2%}**")

else:
    st.info("Bitte lade ein Bild hoch, um die Erkennung zu starten.")
