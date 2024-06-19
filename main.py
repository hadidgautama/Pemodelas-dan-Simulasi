import streamlit as st

st.title("Aplikasi Streamlit Sederhana")
st.write("Selamat datang di aplikasi Streamlit pertama saya!")
st.write("Ini adalah contoh teks yang ditampilkan menggunakan Streamlit.")

# Contoh input
nama = st.text_input("Masukkan nama Anda:")

if st.button("Submit"):
    st.write(f"Hello, {nama}!")
