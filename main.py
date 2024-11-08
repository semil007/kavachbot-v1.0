import os
from pathlib import Path
import streamlit as st
from embeddings import create_or_load_vectorstore
from chatbot import get_kavach_decision
import logging
import fitz  # PyMuPDF for PDF rendering and highlighting
from urllib.parse import urlencode
from googletrans import Translator  # Import googletrans for testing multilingual support

# Set Base Directory
BASE_DIR = Path(__file__).parent

# Configure logging
log_file = BASE_DIR / 'main.log'
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Set Page Configuration
st.set_page_config(
    page_title="Kavach Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load custom CSS
def load_css(file_name):
    css_path = BASE_DIR / file_name
    if css_path.exists():
        with css_path.open() as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logging.info(f"Loaded CSS from '{css_path}'.")
    else:
        st.warning(f"CSS file '{file_name}' not found. Proceeding without custom styles.")
        logging.warning(f"CSS file '{css_path}' not found.")

# Load CSS
load_css("styles.css")

# Title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Kavach Chatbot</h1>", unsafe_allow_html=True)

# Paths
pdf_path = BASE_DIR / 'kavach_guidelines.pdf'
vectorstore_path = BASE_DIR / 'kavach_vectorstore'

# Initialize or load the vectorstore
with st.spinner("Initializing vector store..."):
    try:
        vectorstore, page_images = create_or_load_vectorstore(pdf_path, vectorstore_path)
        logging.info("Vector store and images initialized successfully.")
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        logging.error(f"Initialization failed: {e}")
        st.stop()

# Initialize Conversation History in Session State
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Initialize the Translator
translator = Translator()

# Function to extract a highlighted page image from a PDF
def extract_highlighted_page(pdf_path, page_number, text_to_highlight):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)  # Pages are zero-indexed in PyMuPDF

        # Search for the text and highlight it
        text_instances = page.search_for(text_to_highlight)
        for inst in text_instances:
            page.draw_rect(inst, color=(1, 0, 0), fill=(1, 0.8, 0.8, 0.4))

        # Save page as an image
        image_filename = f"highlighted_page_{page_number}.png"
        image_path = BASE_DIR / image_filename
        pix = page.get_pixmap()
        pix.save(str(image_path))
        logging.info(f"Extracted highlighted page {page_number} as '{image_path}'.")

        return image_path
    except Exception as e:
        logging.error(f"Error extracting highlighted page: {e}")
        return None

# Display Conversation History
def display_chat_history():
    """Function to display the entire conversation history."""
    for idx, msg in enumerate(st.session_state['messages']):
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['content']}")
        elif msg['role'] == 'assistant':
            st.markdown(f"**Chatbot:** {msg['content']}")
            if 'pages' in msg and msg['pages']:
                # Add clickable links for source pages using markdown links
                page_links = []
                for page in msg['pages']:
                    query_params = urlencode({"page": page, "content": msg['content']})
                    link = f"[Page {page}](/?{query_params})"
                    page_links.append(link)
                source_links = ", ".join(page_links)
                st.markdown(f"*Source Pages:* {source_links}", unsafe_allow_html=True)

            # Display images related to the response, if any, in a two-column grid
            if 'images' in msg and msg['images']:
                st.markdown("**Related Images:**")
                num_images = len(msg['images'])
                for i in range(0, num_images, 2):
                    cols = st.columns(2, gap="small")
                    for j in range(2):
                        if i + j < num_images:
                            img_path = BASE_DIR / msg['images'][i + j]
                            if img_path.exists():
                                try:
                                    cols[j].image(str(img_path), width=300)
                                    logging.info(f"Displayed related image: {img_path}")
                                except Exception as e:
                                    cols[j].error(f"Failed to load image: {img_path}")
                                    logging.error(f"Failed to load image '{img_path}': {e}")
                            else:
                                cols[j].error(f"Image not found: {img_path}")
                                logging.warning(f"Image not found: {img_path}")

# Display the conversation history initially
display_chat_history()

# Check if there are query parameters to extract highlighted content
query_params = st.experimental_get_query_params()
if "page" in query_params and "content" in query_params:
    page_number = int(query_params["page"][0])
    content_to_highlight = query_params["content"][0]

    with st.spinner(f"Loading and highlighting page {page_number}..."):
        highlight_image = extract_highlighted_page(pdf_path, page_number, content_to_highlight)
        if highlight_image:
            st.image(str(highlight_image), caption=f"Highlighted Page {page_number}", use_column_width=True)

# User Input at the Bottom using a form
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("You:", key="input")
    language = st.selectbox("Select Language", ["English", "Hindi", "Tamil", "Telugu", "Kannada"])
    submit_button = st.form_submit_button("Send")

    if submit_button:
        if user_query.strip() == "":
            st.warning("Please enter a valid question.")
            logging.warning("User attempted to send an empty query.")
        else:
            # Translate user query to English if needed (only for internal processing)
            user_query_english = user_query
            user_query_translated = user_query  # Initialize with original query
            if language != "English":
                lang_code = {'Hindi': 'hi', 'Tamil': 'ta', 'Telugu': 'te', 'Kannada': 'kn'}
                try:
                    user_query_english = translator.translate(user_query, dest='en').text
                    logging.info(f"User query translated to English: {user_query_english}")
                except Exception as e:
                    logging.error(f"Translation error: {e}")
                    st.warning(f"Translation failed: {e}")

                # Translate the original English input to selected language for display
                try:
                    user_query_translated = translator.translate(user_query, dest=lang_code[language]).text
                except Exception as e:
                    logging.error(f"Translation error: {e}")
                    user_query_translated = user_query  # Fallback to original input

            # Append translated user message to conversation history for display purposes
            st.session_state['messages'].append({'role': 'user', 'content': user_query_translated})
            logging.info(f"User query in selected language: {user_query_translated}")

            # Process the query and get the assistant's response
            with st.spinner("Processing your query..."):
                try:
                    decision, pages, images = get_kavach_decision(vectorstore, user_query_english, page_images)
                    logging.info(f"Assistant decision: {decision}")
                except Exception as e:
                    decision = f"An error occurred while processing your request: {e}"
                    pages = []
                    images = []
                    logging.error(f"Error during get_kavach_decision: {e}")

            # Translate response back to the user's selected language
            decision_translated = decision
            if language != "English":
                try:
                    decision_translated = translator.translate(decision, dest=lang_code[language]).text
                    logging.info(f"Assistant decision translated to {language}: {decision_translated}")
                except Exception as e:
                    logging.error(f"Translation error: {e}")
                    st.warning(f"Translation failed: {e}")
                    decision_translated = decision  # Fallback to original response

            # Append assistant message to conversation history
            st.session_state['messages'].append({
                'role': 'assistant',
                'content': decision_translated,
                'pages': pages,
                'images': [Path(img).relative_to(BASE_DIR) for img in images]  # Ensure paths are relative
            })

            # Clear query parameters and update conversation history display
            st.experimental_set_query_params()  # Clear query parameters
            st.empty()  # Clear any existing content
            display_chat_history()

# Optional: Hide Streamlit's default footer, hamburger menu, and warning boxes for a cleaner interface
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stAlert {display: none;}  /* Hide warning boxes */
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)