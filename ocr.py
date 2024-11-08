import os
import easyocr
import logging
import joblib
from embedding_handler import get_huggingface_embeddings

# Configure logging
logging.basicConfig(
    filename='ocr.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Get HuggingFace Embeddings
embeddings = get_huggingface_embeddings()

def extract_text_and_embeddings(image_path):
    if not os.path.exists(image_path):
        logging.error(f"Image file '{image_path}' not found.")
        return "", None

    try:
        results = reader.readtext(image_path, detail=0)
        text = ' '.join(results).strip()
        logging.info(f"Extracted text from '{image_path}': {text[:100]}...")

        if text:
            text_embedding = embeddings.embed_query(text)
            return text, text_embedding
        else:
            return "", None
    except Exception as e:
        logging.error(f"Error extracting text from '{image_path}': {e}")
        return "", None

def save_embeddings(embedding, embedding_path):
    try:
        joblib.dump(embedding, embedding_path)
        logging.info(f"Saved embeddings to '{embedding_path}'.")
    except Exception as e:
        logging.error(f"Error saving embeddings to '{embedding_path}': {e}")

def load_embeddings(embedding_path):
    try:
        embedding = joblib.load(embedding_path)
        logging.info(f"Loaded embeddings from '{embedding_path}'.")
        return embedding
    except Exception as e:
        logging.error(f"Error loading embeddings from '{embedding_path}': {e}")
        return None