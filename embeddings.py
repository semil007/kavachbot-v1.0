import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from embedding_handler import get_huggingface_embeddings
from image_extractor import extract_images
import logging
import streamlit as st

# Configure logging
logging.basicConfig(
    filename='embeddings.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

@st.cache_resource(show_spinner=False)
def create_or_load_vectorstore(pdf_path, vectorstore_path, images_dir='extracted_images', metadata_file='image_metadata.json'):
    embeddings = get_huggingface_embeddings()
    logging.info("Initialized HuggingFaceEmbeddings.")

    # Extract images and map them to pages if metadata is not already saved
    logging.info("Starting image extraction or loading metadata.")
    page_images = extract_images(pdf_path, images_dir, metadata_file=metadata_file)
    logging.info(f"Extracted images from {len(page_images)} pages.")
    
    # Load vectorstore if it exists
    if os.path.exists(vectorstore_path):
        logging.info("Loading existing vector store.")
        try:
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            logging.info("Vector store loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            raise e
    else:
        logging.info("Vector store not found. Creating a new one.")
        try:
            pdf_reader = PdfReader(pdf_path)
            logging.info(f"Opened PDF file '{pdf_path}' for reading.")
        except Exception as e:
            logging.error(f"Error reading PDF: {e}")
            raise e

        page_texts = {}
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                page_texts[page_num] = page_text

        # Updated Text Splitting Strategy
        chunk_size = 500  # Reduced chunk size to maintain better control over context.
        chunk_overlap = 150  # Increased overlap to maintain continuity between chunks.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        kavach_chunks = []

        for page, text in page_texts.items():
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                kavach_chunks.append({'text': chunk, 'source': 'kavach_source', 'page': page})

        logging.info(f"Total text chunks created: {len(kavach_chunks)}")

        try:
            vectorstore = FAISS.from_texts(
                texts=[chunk['text'] for chunk in kavach_chunks],
                embedding=embeddings,
                metadatas=[{'source': chunk['source'], 'page': chunk['page']} for chunk in kavach_chunks]
            )
            logging.info("Vector store created successfully.")
        except Exception as e:
            logging.error(f"Error creating vector store: {e}")
            raise e

        # Save the vectorstore
        try:
            vectorstore.save_local(vectorstore_path)
            logging.info("Vector store saved successfully.")
        except Exception as e:
            logging.error(f"Error saving vector store: {e}")
            raise e
    
    return vectorstore, page_images