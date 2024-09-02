from langchain_google_genai import GoogleGenerativeAIEmbeddings
import getpass
import os
import google.generativeai as genai
google_api_key="GOOGLE_API_KEY"
def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    return embeddings
