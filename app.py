
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageOps

import streamlit as st
from streamlit_option_menu import option_menu

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

from werkzeug.utils import secure_filename


# load model
model = load_model("ModifiedCNN.h5")



st.write("""
         # Cetacean Photo Identification
         """
         )

st.write("Machine Learning and Image Processing Methods for Cetacean Photo Identification: A Systematic Review")


#if selection == "Home":
with st.expander("Home"):

    st.title("Machine Learning and Image Processing Methods for Cetacean Photo Identification: A Systematic Review")
    homelogo = Image.open(r'css/cetacean_photo.jpg')
    st.image(homelogo, width=700)
    
#if selection == "Performace":
with st.expander("Performance"):

    st.title("Performance on Cetacean Photo Identification")
    st.markdown("<h1 style='text-align: center; font-size: 20px; color: blue;'>Modified Convolutional Neural Network Architecture</h1>"
                , unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 20px; color: lime;'>Accuracy % Score: 99.8</h1>"
                , unsafe_allow_html=True)
    performaceimg = Image.open(r'css/training_graph.png')
    st.image(performaceimg, width=700)


file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
  

def transform_img(file_path):
    print(file_path)
    img = image.load_img(file_path, target_size=(224, 224))

    # Preprocessing the image
    x = np.array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    return x

    
# Main function:
def main():     
    if file is None:
    
        st.text("No Input Given? Please upload an image file")
    
    else:
        uploadedImage = Image.open(file)
        uploadedImage = uploadedImage.resize((600, 400))
        st.image(uploadedImage, use_column_width=True)
        # Save the file path from ./test
        basepath = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(basepath, 'test', secure_filename(file.name))

        transform_img_data = transform_img(file_path)

        if st.sidebar.button("submit"):
            prediction = model.predict(transform_img_data)
            # st.success(prediction)
            preds=np.argmax(prediction, axis=1)
            # st.success(preds)
        
            if np.argmax(prediction) == 0:
                st.sidebar.success("\n Detected: Marine Fissipeds")
            elif np.argmax(prediction) == 1:
                st.sidebar.success("\n Detected: Pinnipeds")
            elif np.argmax(prediction) == 2:
                st.sidebar.success("\n Detected: Cetacean Detected")
    
    
if __name__ == '__main__':
    main()
            
    
        