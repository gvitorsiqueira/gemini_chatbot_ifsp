from langchain_google_genai import GoogleGenerativeAIEmbeddings
import getpass
import os
import google.generativeai as genai
import streamlit as st
google_api_key=st.secrets["GOOGLE_API_KEY"]
def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)
    return embeddings
