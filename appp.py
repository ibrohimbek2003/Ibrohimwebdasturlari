import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import numpy as np
import os

# PosixPath muammosini hal qilish
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Model yo'qligini tekshirish
if not os.path.exists("transport_model.pkl"):
    st.error("Model fayli topilmadi. Faylni yuklang va yo'lni tekshiring!")
else:
    # Rasm yuklash
    uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Yuklangan rasmni o'qish
        img = PILImage.create(uploaded_file)

        # Modelni yuklash
        learner = load_learner("transport_model.pkl")

        # Debugging: Learner turini tekshirish
        st.write(f"Learner turi: {type(learner)}")

        # Rasmni aniqlash
        try:
            # Learner obyektining to'g'ri ekanligini tekshirish
            if isinstance(learner, Learner):
                pred, pred_idx, probs = learner.predict(img)

                # Natijani ko'rsatish
                st.image(np.array(img), caption='Yuklangan rasm', use_column_width=True)
                st.write(f"Bu rasm: {pred} (Ishonch: {probs[pred_idx]:.2f})")
            else:
                st.error("Learner obyektida xato. Model yuklash jarayonini tekshiring.")
        except Exception as e:
            st.error(f"Xato yuz berdi: {e}")
