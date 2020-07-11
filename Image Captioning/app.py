import streamlit as st
from PIL import Image
import string
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras
import sys, time, os, warnings 
warnings.filterwarnings("ignore")
import re
import numpy as np
import pandas as pd 
from PIL import Image
import pickle
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16, preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing import sequence
from keras.models import load_model
from keras.preprocessing import image

#=================================================================#===============================================#===============


#=============Model/Predictions=========================#


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
def preprocess(img):
    #img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image)
    temp_enc = incep_model.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

def greedySearch(photo):
    in_text = 'start'
    for i in range(max_len):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_len)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        
        in_text += ' ' + word
        if word == 'end':
            break
    final = in_text.split()
    final = final[1:]
    final = ' '.join(final)
    return final


#=======================================================#


@st.cache
def load_image(img):
    im=Image.open(img)
    return im

def main():
    st.title('Image Captioning')
    #st.text('built')
    activities=["caption","About"]
    choice=st.sidebar.selectbox("select activity", activities)
    
    if choice=="caption":
        st.subheader("caption")
        model = load_model('my_model.h5')
        incep_model = load_model('incep_model.h5')
        ixtoword = load(open('indextoword.pkl', 'rb'))
        wordtoix = load(open('wordtoindex.pkl', 'rb'))
        max_len=40

        image_file=st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
        
        if image_file is not None:
            our_image=Image.open(image_file)
            new_img = our_image.resize((299,299))
            st.text('Original Image')
            st.image(new_img)
            
        if st.button("Process"):
            new_img = our_image.resize((299,299))
            imagex = encode(new_img).reshape((1,2048))
            st.text("Greedy:"+greedySearch(imagex))
    else:
        st.subheader("About")
    
    
if __name__=='__main__':
    main()