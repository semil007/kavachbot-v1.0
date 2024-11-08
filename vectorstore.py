import os
from pathlib import Path
import joblib
from joblib import load
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from image_extractor import extract_images
import logging

# Set Base Directory
BASE_DIR = Path(__file__).parent

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Configure logging
log_file = BASE_DIR / 'vectorstore.log'
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def create_or_load_vectorstore(pdf_path, vectorstore_path):
    pdf_path = Path(pdf_path)
    vectorstore_path = Path(vectorstore_path)
    images_dir = BASE_DIR / 'extracted_images'
    metadata_file = BASE_DIR / 'image_metadata.json'

    # Initialize page_images as an empty dictionary to ensure it has a value
    page_images = {}

    try:
        # Check if vectorstore exists
        index_file = vectorstore_path / 'index.joblib'
        if index_file.exists():
            logging.info(f"Loading existing vector store from '{index_file}'...")
            vectorstore = load(str(index_file))
        else:
            # Extract text and images from the PDF
            logging.info(f"Extracting text and images from '{pdf_path}'...")
            page_images = extract_images(pdf_path, images_dir, metadata_file=metadata_file)

            # Create new vector store
            texts = [image['ocr_text'] for page in page_images.values() for image in page if image['ocr_text']]
            metadatas = [{'source': f'Page {page_num}'} for page_num, page in page_images.items() for image in page if image['ocr_text']]
            
            if not texts:
                raise Exception("No text extracted from images to create vector store.")

            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas
            )

            # Save vector store locally using joblib
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(vectorstore, str(index_file))
            logging.info(f"Vector store saved to '{index_file}'.")
    
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        vectorstore = None  # Set vectorstore to None in case of failure

    return vectorstore, page_images