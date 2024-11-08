import os
from pathlib import Path
import google.generativeai as genai
import logging
import pickle
import numpy as np
from embedding_handler import get_huggingface_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import time

# Set Base Directory
BASE_DIR = Path(__file__).parent

# Configure logging
log_file = BASE_DIR / 'chatbot.log'
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize HuggingFace Embeddings
embeddings = get_huggingface_embeddings()

# Set up Google Gemini API
api_key = os.getenv("GOOGLE_GEMINI_PRO_API_KEY")
if not api_key:
    raise ValueError("Please set the GOOGLE_GEMINI_PRO_API_KEY environment variable.")
genai.configure(api_key=api_key)

# Simple cache for responses
response_cache = {}

def sanitize_response(response):
    """
    Removes all HTML tags from the response using BeautifulSoup.
    """
    soup = BeautifulSoup(response, "html.parser")
    return soup.get_text()

def generate_kavach_response(prompt, max_retries=3):
    """
    Generates a response using the Google Gemini API, with retry logic and caching.
    """
    # Check if response is cached
    if prompt in response_cache:
        logging.info("Returning cached response.")
        return response_cache[prompt]

    # Implement exponential backoff
    wait_time = 2  # Initial wait time in seconds
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-1.5-pro-002')
            response = model.generate_content(prompt)
            logging.info("Successfully generated response from Gemini Pro API.")
            response_text = response.text
            # Cache the response
            response_cache[prompt] = response_text
            return response_text
        except Exception as e:
            logging.error(f"API call error: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {wait_time} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                wait_time *= 2  # Double the wait time for each retry
            else:
                return "Error generating response: Resource has been exhausted or an unexpected error occurred. Please try again later."

def get_kavach_decision(vectorstore, query, page_images):
    """
    Retrieves relevant documents and images based on the user's query and generates an appropriate response.
    """
    try:
        logging.info(f"Received query: {query}")

        # Retrieve relevant chunks from the vector store
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Increased `k` to get more relevant context.
        relevant_docs = retriever.invoke(query)
        logging.info(f"Number of relevant documents retrieved: {len(relevant_docs)}")

        if not relevant_docs:
            logging.warning("No relevant documents found for the query.")
            return "I'm sorry, I couldn't find relevant information to answer your question.", [], []

        # Extract unique page numbers from the relevant documents
        pages = set()
        combined_texts = []
        for doc in relevant_docs:
            page = doc.metadata.get('page')
            if page:
                pages.add(page)
            combined_texts.append(doc.page_content)

        combined_text = "\n".join(combined_texts)
        logging.debug(f"Combined text for prompt: {combined_text[:500]}...")

        # Embed the query to compare with image embeddings
        query_embedding = embeddings.embed_query(query)
        relevant_images = []

        # Compare query embedding with OCR embeddings of images on the relevant pages
        for page in pages:
            if str(page) in page_images:
                for image_data in page_images[str(page)]:
                    embedding_path = Path(image_data.get('embedding_path'))
                    image_path = Path(image_data['path'])

                    if embedding_path is None or not embedding_path.exists():
                        logging.warning(f"Embedding file '{embedding_path}' not found or missing. Skipping image '{image_path}'.")
                        continue

                    try:
                        with embedding_path.open('rb') as f:
                            image_embedding = pickle.load(f)

                        # Compute cosine similarity
                        similarity = cosine_similarity([query_embedding], [image_embedding])[0][0]
                        logging.info(f"Similarity score for image '{image_path}': {similarity}")

                        # Lowering the similarity threshold to include more relevant images 
                        if similarity > 0.22:  # .22 is a placeholder threshold
                            relevant_images.append(str(image_path))
                    except Exception as e:
                        logging.error(f"Error loading embedding from '{embedding_path}': {e}")
                        continue

        # Formulate the final prompt for the Gemini model with clear instructions
        prompt = (
            f"Based on the following Kavach guidelines:\n\n{combined_text}\n\n"
            f"Answer the query: {query}\n\n"
            "Please provide a clear and concise answer in plain text without any HTML or markdown tags."
        )

        # Generate the decision using the Gemini model
        decision = generate_kavach_response(prompt)
        logging.info(f"Generated decision: {decision}")

        # Sanitize the decision to remove any HTML tags
        decision = sanitize_response(decision)

        return decision, sorted(pages), relevant_images
    except Exception as e:
        logging.error(f"Error processing decision: {e}")
        return f"Error processing your request: {e}", [], []