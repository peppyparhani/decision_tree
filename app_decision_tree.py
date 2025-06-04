import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    df = pd.read_csv("data_balita.csv")

    # Encoder untuk Jenis Kelamin
    le_gender = LabelEncoder()
    df['Jenis Kelamin'] = le_gender.fit_transform(df['Jenis Kelamin'])

    # Encoder untuk Status Gizi
    le_status = LabelEncoder()
    df['Status Gizi'] = le_status.fit_transform(df['Status Gizi'])

    # Fitur dan Label
    X = df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
    y = df['Status Gizi']

    model = DecisionTreeClassifier(criterion="gini", splitter='best')
    model.fit(X, y)

    return model, le_status, le_gender

model, le_status, le_gender = load_model()

# ---------------------------
# UI Streamlit
# ---------------------------
st.title("🌳 Deteksi Stunting Balita dengan Decision Tree")

umur = st.number_input("Umur Balita (bulan)", min_value=0, max_value=60, value=24)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"])
tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, value=80.0)

if st.button("Prediksi"):
    jk_encoded = le_gender.transform([jenis_kelamin])[0]
    input_data = [[umur, jk_encoded, tinggi]]

    hasil_encoded = model.predict(input_data)[0]
    hasil = le_status.inverse_transform([hasil_encoded])[0]

    st.success(f"🌟 Status Gizi Balita: **{hasil.upper()}**")
