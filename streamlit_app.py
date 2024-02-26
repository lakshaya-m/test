#pip install matplotlib

import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
from zipfile import ZipFile
  
# specifying the zip file name
file_name = "best_plant_model.zip"
  
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    zip.extractall()


st.title('PLANT DISEASE DETECTION')
st.sidebar.title("PLANT DISEASE DETECTION")
st.sidebar.caption("Know The Disease Affecting Our Flora")
st.sidebar.markdown("Made by LAKSHAYA M, MSC IT, Annamalai University")
st.sidebar.caption("This model is trained with 87k plant diseased leaves ")
st.sidebar.markdown("'Tomato', 'Grape', 'Orange', 'Soyabean', 'Squash', 'Potato', 'Corn_(maize)', 'Strawberry', 'Peach', 'Apple', 'Blueberry', 'Cherry_(including_sour)', 'Pepper,_bell', 'Raspberry'")


page_bg_img = '''
<style>
body {
background-image: url(r"image.jpeg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

model = tf.keras.models.load_model('best_plant_model.h5')

class_names = ['Apple___Apple_scab',
 'Apple_Black_rot',
 'Apple_Cedar_apple_rust',
 'Apple_healthy',
 'Blueberry_healthy',
 'Cherry_(including_sour)_Powdery_mildew',
 'Cherry_(including_sour)_healthy',
 'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)_Common_rust_',
 'Corn_(maize)_Northern_Leaf_Blight',
 'Corn_(maize)_healthy',
 'Grape___Black_rot',
 'Grape_Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

def load_prep(img_path):
  img = tf.io.read_file(img_path)
  img = tf.image.decode_image(img)
  img = tf.image.resize(img,size=(224,224))
  return img


def predict_img(img,model=model):
  # img = load_prep(img_path)

  pred = model.predict(tf.expand_dims(img,axis=0))

  pred_name = class_names[pred.argmax()]

  return pred_name

uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
if uploaded_file is not None:
	image = mpimg.imread(uploaded_file)
	st.image(image, caption='Given image')
	img1 = tf.image.resize(image,(224,224))
	st.write('# Prediction for the given image:')
	st.write(predict_img(img1,model))


