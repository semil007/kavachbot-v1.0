from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path

def get_huggingface_embeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    return HuggingFaceEmbeddings(model_name=model_name)