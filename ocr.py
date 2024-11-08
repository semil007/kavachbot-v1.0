import os
from pathlib import Path
import easyocr
import logging
import joblib
from embedding_handler import get_huggingface_embeddings

# Set Base Directory
BASE_DIR = Path(__file__).parent

# Configure logging
log_file = BASE_DIR / 'ocr.log'
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Get HuggingFace Embeddings
embeddings = get_huggingface_embeddings()

def extract_text_and_embeddings(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        logging.error(f"Image file '{image_path}' not found.")
        return "", None

    try:
        results = reader.readtext(str(image_path), detail=0)
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
    embedding_path = Path(embedding_path)
    try:
        with embedding_path.open('wb') as f:
            joblib.dump(embedding, f)
        logging.info(f"Saved embeddings to '{embedding_path}'.")
    except Exception as e:
        logging.error(f"Error saving embeddings to '{embedding_path}': {e}")

def load_embeddings(embedding_path):
    embedding_path = Path(embedding_path)
    try:
        with embedding_path.open('rb') as f:
            embedding = joblib.load(f)
        logging.info(f"Loaded embeddings from '{embedding_path}'.")
        return embedding
    except Exception as e:
        logging.error(f"Error loading embeddings from '{embedding_path}': {e}")
        return None