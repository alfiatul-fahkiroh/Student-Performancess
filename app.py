# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model_graduation.pkl')

# Judul aplikasi
st.title("Aplikasi Prediksi Kategori Waktu Kelulusan Mahasiswa")

st.write("""
Masukkan data berikut untuk memprediksi kategori masa studi:
""")

# Form input
ACT = st.number_input("Masukkan nilai ACT composite score:", min_value=0.0, step=0.1)
SAT = st.number_input("Masukkan nilai SAT total score:", min_value=0.0, step=0.1)
GPA = st.number_input("Masukkan nilai rata-rata SMA:", min_value=0.0, step=0.01)
income = st.number_input("Masukkan nilai pendapatan orang tua:", min_value=0.0, step=100.0)
education = st.text_input("Masukkan tingkat pendidikan orang tua (teks):")

# Tombol Prediksi
if st.button("Prediksi"):
    try:
        # Buat DataFrame dari input user
        new_data_df = pd.DataFrame([[ACT, SAT, GPA, income, education]],
                                   columns=['ACT composite score',
                                            'SAT total score',
                                            'high school gpa',
                                            'parental income',
                                            'parent_edu_numerical'])
        
        # Lakukan prediksi
        prediction = model.predict(new_data_df)
        
        st.success(f"Prediksi kategori masa studi: {prediction[0]}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
