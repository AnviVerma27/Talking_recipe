import streamlit as st
import joblib
import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import helper as help
from elevenlabs import generate, play
import json

st.set_page_config(page_title ="TalkingRecipeBook", layout='centered')

@st.cache_resource
def load_model():
    return joblib.load("transformer.joblib")
def load_embeddings():
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = joblib.load("embeddings.pkl")
    return st.session_state.embeddings

def load_sentences():
    if "sentences" not in st.session_state:
        st.session_state.sentences = joblib.load("sentences.pkl")
    return st.session_state.sentences

def load_dataset():
    if "dataset" not in st.session_state:
        st.session_state.dataset = pd.read_parquet('final_recipes.parquet')
    return st.session_state.dataset

model = load_model()
embeddings = load_embeddings()
sentences = load_sentences()



help.set_bg_hack()

df = load_dataset()

title="""<center> Talking recipe book </center>"""

help.header1(title)

text = """
    <center> <br> Welcome to the talking recipe book. </br> </center>
    <center> This app will help you in making new dishes everyday.</center>
    <center> It has an inbuilt voice feature that will recite the whole recipe to you for your ease</center>
    """
help.header3(text)

help.header2("Enter the dish.....")

paper_name = st.text_input("Input here")

def predict():
    encoded_name = model.encode(paper_name)

    cosine_scores = cosine_similarity([encoded_name], embeddings)
    top_dish = np.argsort(cosine_scores, axis=1)[0][-1:][::-1]
    
    for i in top_dish:
        help.header2(df[df['title'] == sentences[i]]['title'].values[0])
        str=df[df['title'] == sentences[i]]['directions'].values[0]
        s=str.replace('[','')
        t=s.replace(']','')
        str1=t.replace('"','')
        list=str1.split(',')
        paragraph_length = 1
        array = json.loads(df[df['title'] == sentences[i]]['ingredients'].values[0])
        help.header3("Ingredients-")
        for element in array:
            st.write(element)
        
        help.header3("Recipe-")
        paragraphs = [list[i:i+paragraph_length] for i in range(0, len(list), paragraph_length)]
        for paragraph in paragraphs:
            paragraph_text = ','.join(paragraph)
            st.write(paragraph_text)
            text_to_speech(paragraph_text)

def text_to_speech(text):
    audio = generate(
    text=text,
    voice="Bella",
    model='eleven_monolingual_v1',
    api_key='78e907ae7b2a82979ff65e285400bf07'
    )
    play(audio)

if st.button('Display'):
    predict()
