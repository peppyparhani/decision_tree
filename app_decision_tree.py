import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Load dan Siapkan Model
# ---------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("data_balita.csv")

    # Label encoding untuk semua kolom (agar cocok dengan notebook)
    le = LabelEncoder()
    df_encoded = df.apply(le.fit_transform)

    X = df_encoded[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
    y = df_encoded['Status Gizi']

    model = DecisionTreeClassifier(criterion="gini", splitter='best')
    model.fit(X, y)

    # Simpan label encoder untuk hasil prediksi
    le_status = LabelEncoder()
    le_status.fit(df['Status Gizi'])

    # Mapping jenis kelamin manual (karena LabelEncoder bisa beda urutan)
    gender_map = {'laki-laki': df['Jenis Kelamin'].unique()[0], 'perempuan': df['Jenis Kelamin'].unique()[1]}
    gender_enc = {v: le.transform([v])[0] for v in gender_map.values()}

    return model, le_status, gender_enc

model, le_status, gender_enc = load_model()

# ---------------------------
# UI Streamlit
# ---------------------------
st.title("🌳 Deteksi Stunting Balita dengan Decision Tree")

umur = st.number_input("Umur Balita (bulan)", min_value=0, max_value=60, value=24)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"])
tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, value=80.0)

if st.button("Prediksi"):
    jk_encoded = gender_enc[jenis_kelamin]
    input_data = [[umur, jk_encoded, tinggi]]

    hasil_encoded = model.predict(input_data)[0]
    hasil = le_status.inverse_transform([hasil_encoded])[0]

    st.success(f"🌟 Status Gizi Balita: **{hasil.upper()}**")
