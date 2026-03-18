import streamlit as st



st.title("Plant Detection App")

st.markdown(f"### Upload an image of a plant to detect its type.")

st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

st.markdown(f"### Or take a picture with your camera:")
st.camera_input("Take a picture")