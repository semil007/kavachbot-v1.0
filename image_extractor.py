import os
from pathlib import Path
import fitz  # PyMuPDF
import logging
import json
from ocr import extract_text_and_embeddings, save_embeddings

# Set Base Directory
BASE_DIR = Path(__file__).parent

# Configure logging
log_file = BASE_DIR / 'image_extractor.log'
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def extract_images(pdf_path, images_dir='extracted_images', embeddings_dir='image_embeddings', metadata_file='image_metadata.json'):
    pdf_path = Path(pdf_path)
    images_dir = BASE_DIR / images_dir
    embeddings_dir = BASE_DIR / embeddings_dir
    metadata_file = BASE_DIR / metadata_file

    # If metadata file exists, load it and return
    if metadata_file.exists():
        logging.info(f"Loading image metadata from '{metadata_file}'.")
        with metadata_file.open('r') as f:
            page_images = json.load(f)
        return page_images

    # Otherwise, extract images and save metadata
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
        logging.info(f"Created images directory at '{images_dir}'.")

    if not embeddings_dir.exists():
        embeddings_dir.mkdir(parents=True)
        logging.info(f"Created embeddings directory at '{embeddings_dir}'.")

    try:
        doc = fitz.open(str(pdf_path))
        logging.info(f"Opened PDF file '{pdf_path}'.")
    except Exception as e:
        logging.error(f"Error opening PDF file '{pdf_path}': {e}")
        raise e

    page_images = {}
    total_images = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)
        image_data = []

        if images:
            logging.info(f"Found {len(images)} image(s) on page {page_num + 1}.")

        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
            except Exception as e:
                logging.error(f"Error extracting image {img_index + 1} on page {page_num + 1}: {e}")
                continue

            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            image_path = images_dir / image_filename

            # Skip if image already exists
            if image_path.exists():
                logging.info(f"Image '{image_filename}' already exists. Skipping extraction.")
            else:
                try:
                    with image_path.open("wb") as img_file:
                        img_file.write(image_bytes)
                    logging.info(f"Saved image as '{image_filename}'.")
                except Exception as e:
                    logging.error(f"Error saving image {img_index + 1} on page {page_num + 1}: {e}")
                    continue

            # Extract OCR text and generate embeddings
            try:
                ocr_text, ocr_embedding = extract_text_and_embeddings(str(image_path))

                if ocr_embedding is not None:
                    embedding_filename = f"page_{page_num + 1}_img_{img_index + 1}_embedding.joblib"
                    embedding_path = embeddings_dir / embedding_filename
                    if not embedding_path.exists():
                        save_embeddings(ocr_embedding, str(embedding_path))
                        logging.info(f"Saved embedding for '{image_filename}' to '{embedding_path}'.")
                else:
                    logging.warning(f"No embedding generated for '{image_filename}'. OCR text might be empty or invalid.")
                    embedding_path = None
            except Exception as e:
                logging.error(f"Error during OCR or embedding generation for '{image_filename}': {e}")
                embedding_path = None

            # Append image details to the image data
            image_data.append({
                'path': str(image_path),
                'ocr_text': ocr_text,
                'embedding_path': str(embedding_path) if ocr_embedding is not None else None
            })
            total_images += 1

        if image_data:
            page_images[str(page_num + 1)] = image_data

    logging.info(f"Total images extracted: {total_images}")

    # Save metadata to JSON file
    with metadata_file.open('w') as f:
        json.dump(page_images, f)
    logging.info(f"Saved image metadata to '{metadata_file}'.")

    return page_images