import streamlit as st
import joblib
import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = joblib.load('transformer.joblib')
embeddings = joblib.load('embeddings.pkl')
sentences = joblib.load('sentences.pkl')

df = pd.read_parquet('final_recipes.parquet')

st.title('Enter the Dish...')
paper_name = st.text_input("Input here")

def predict():
    encoded_name = model.encode(paper_name)

    cosine_scores = cosine_similarity([encoded_name], embeddings)
    top_similar_dishes = np.argsort(cosine_scores, axis=1)[0][-5:][::-1]
    top_dish = np.argsort(cosine_scores, axis=1)[0][-1:][::-1]

    for i in top_dish:
        st.write(df[df['title'] == sentences[i]]['directions'].values[0])

if st.button('Display'):
    predict()