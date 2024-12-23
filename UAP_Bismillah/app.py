import streamlit as st
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib

# Load model
fnn_model = tf.keras.models.load_model("depression_fnn_model.keras")
xgb_model = joblib.load("depression_xgboost_model.joblib")  # Pastikan file model XGBoost benar

# Judul aplikasi
st.title("Prediksi Risiko Depresi Mahasiswa")
st.write("Masukkan data berikut untuk memprediksi risiko depresi.")

model_choice = st.radio("Pilih Model Prediksi", ["FNN", "Random Forest"])

# Input data pengguna
age = st.slider("Usia", 18, 60, 25)
cgpa = st.slider("Nilai CGPA", 0.0, 10.0, 5.0, step=0.1)
academic_pressure = st.slider("Tekanan Saat Belajar (0-5)", 0, 5, 1)
work_pressure = st.slider("Tekanan Saat Kerja (0-5)", 0, 5, 1)
study_satisfaction = st.slider("Kepuasan Belajar (0-1)", 0.0, 1.0, 0.5)
job_satisfaction = st.slider("Kepuasan Kerja (0-1)", 0.0, 1.0, 0.5)
sleep_duration = st.selectbox("Durasi Tidur", ['5-6 jam', 'Kurang dari 5 jam', '7-8 jam', 'More than 8 jam', 'Others'])
work_study_hours = st.slider("Jam Kerja/Belajar per Hari", 0, 12, 8)
financial_stress = st.slider("Stres Finansial (0-1)", 0.0, 1.0, 0.5)

# Data kategorikal
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
degree = st.selectbox("Jenjang Pendidikan", ["SMA", "Sarjana", "Magister", "Doktor", "Lainnya"])
dietary_habits = st.selectbox("Kebiasaan Diet", ["Sehat",  "Biasa", "Lainnya", "Tidak Sehat"])
suicidal_thoughts = st.selectbox("Pernah Memiliki Pikiran Bunuh Diri?", ["Ya", "Tidak"])
family_history = st.selectbox("Riwayat Keluarga dengan Penyakit Mental", ["Ya", "Tidak"])

# Mapping data ke format numerik
gender_mapping = {"Laki-laki": 1, "Perempuan": 0}
degree_mapping = {"SMA":3, "Sarjana": 4, "Magister": 2, "Doktor": 0, "Lainnya": 1}
sleep_duration_mapping = {"5-6 jam": 0, "7-8 jam": 1, "Kurang dari 5 jam": 2, "Lebih dari 8 jam": 3, "Lainnya": 4}
dietary_mapping = {"Sehat": 0, "Biasa": 1, "Lainnya": 2, "Tidak Sehat": 3}
binary_mapping = {"Ya": 1, "Tidak": 0}

# Buat array input
input_data = np.array([[
    gender_mapping[gender],
    age / 60,  # Normalisasi usia
    financial_stress,
    work_study_hours / 12,  # Normalisasi
    cgpa / 10,  # Normalisasi CGPA
    academic_pressure / 5,  # Normalisasi tekanan belajar
    work_pressure / 5, 
    study_satisfaction,
    job_satisfaction,
    dietary_mapping[dietary_habits],
    binary_mapping[suicidal_thoughts],
    binary_mapping[family_history],
    degree_mapping[degree],
    sleep_duration_mapping[sleep_duration]
]])

# Prediksi ketika tombol diklik
if st.button("Prediksi"):
    if model_choice == "FNN":
        prediction_prob = fnn_model.predict(input_data)
        prediction = (prediction_prob > 0.5).astype("int32")
        st.write(f"Model: FNN")
    elif model_choice == "Random Forest":
        prediction_prob = xgb_model.predict(input_data)
        prediction = (prediction_prob > 0.5).astype("int32")
        st.write(f"Model: Random Forest")
        
    st.write(f"Probabilitas Risiko Depresi: {prediction_prob[0][0]:.2f}")
    if prediction[0][0] == 1:
        st.error("Model memprediksi Anda berisiko mengalami depresi.")
    else:
        st.success("Model memprediksi Anda tidak berisiko mengalami depresi.")
