import os
import joblib
from joblib import load
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from image_extractor import extract_images
import logging

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Configure logging
logging.basicConfig(
    filename='vectorstore.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def create_or_load_vectorstore(pdf_path, vectorstore_path):
    # Initialize page_images as an empty dictionary to ensure it has a value
    page_images = {}

    try:
        # Check if vectorstore exists
        if os.path.exists(vectorstore_path):
            logging.info(f"Loading existing vector store from '{vectorstore_path}/index.joblib'...")
            vectorstore = load(os.path.join(vectorstore_path, 'index.joblib'))
        else:
            # Extract text and images from the PDF
            logging.info(f"Extracting text and images from '{pdf_path}'...")
            page_images = extract_images(pdf_path)

            # Create new vector store
            texts = [image['ocr_text'] for page in page_images.values() for image in page if image['ocr_text']]
            metadatas = [{'source': f'Page {page_num}'} for page_num, page in page_images.items() for image in page if image['ocr_text']]
            
            if not texts:
                raise Exception("No text extracted from images to create vector store.")

            vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

            # Save vector store locally using joblib
            if not os.path.exists(vectorstore_path):
                os.makedirs(vectorstore_path)
            joblib.dump(vectorstore, os.path.join(vectorstore_path, 'index.joblib'))
            logging.info(f"Vector store saved to '{vectorstore_path}/index.joblib'.")

    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        vectorstore = None  # Set vectorstore to None in case of failure

    return vectorstore, page_images