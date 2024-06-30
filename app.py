import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from PIL import Image


img=Image.open('movie.jpg')
st.set_page_config(page_title='Moview Review Sentiments',page_icon=img)


model=pk.load(open('model.pkl','rb'))
scaler=pk.load(open('scaler.pkl','rb'))
title = '<h1 style="font-family:sans-serif; color:white;">Sentiment Prediction</h1>'
st.markdown(title, unsafe_allow_html=True)
new_title = '<p style="font-family:sans-serif; color:pink; font-size: 19px;">Enter the movie review:</p>'
st.markdown(new_title, unsafe_allow_html=True)
#st.header('Enter the movie review:')
review=st.text_input(' ')

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://static.vecteezy.com/system/resources/previews/006/852/817/non_2x/abstract-colorful-gradient-lines-with-blue-and-pink-light-on-purple-background-free-vector.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

if st.button('Predict'):
    rev=scaler.transform([review]).toarray()
    result=model.predict(rev)
    if result[0]==0:
        pred = '<p style="font-family:sans-serif; color:pink; font-size: 19px;">Negative Response</p>'
        st.markdown(pred, unsafe_allow_html=True)
    else:
        pred = '<p style="font-family:sans-serif; color:pink; font-size: 19px;">Positive Response</p>'
        st.markdown(pred, unsafe_allow_html=True)