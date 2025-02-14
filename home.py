import streamlit as st
import numpy as np
from PIL import Image
import pickle as pkl




def preprocess_image(image_path):
    
    img = Image.open(image_path).convert('L')  
    img = img.resize((28, 28))  
    img_array = np.array(img)  
    img_array = 255 - img_array  
    img_array = img_array / 255.0 
    img_array = img_array.reshape(1, 28, 28, 1)  
    return img_array

with open("cnn_model.pkl", "rb") as file:
        model = pkl.load(file)

st.title("Digit Recogonizer")
st.title("Made by CODE CRAFT")

st.subheader("Upload image")
img = st.file_uploader("1", label_visibility="collapsed")
if img is not None:
    st.image(img,width=128)
if st.button("Submit"):
    if img is not None:
        img = preprocess_image(img)
        prediction = model.predict(img)
        predict = int(np.argmax(prediction))
        st.subheader(f'Prediction: {predict}')
        print("ALL RIGHTS ARE RESERVED BY CODE CRAFT")
        print("MADE BY CODE CRAFT")
    else:
        st.subheader(f'Prediction: Null')
st.title("ALL RIGHTS ARE RESERVED BY CODE CRAFTE")
