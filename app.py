# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 01:27:53 2023

@author: DaivikM
"""

import streamlit as st
import pickle
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.tokenize import TreebankWordTokenizer  # Using safer tokenizer

# Create and use a local nltk_data folder in a portable way
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download necessary resources
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

ps = PorterStemmer()
tokenizer = TreebankWordTokenizer()

def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)  # Replaces nltk.word_tokenize to avoid punkt issues

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App UI
st.title("Email/SMS Text Classifier")

input_sms = st.text_area("Enter your message:")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display result
    st.header("Spam" if result == 1 else "Not Spam")
