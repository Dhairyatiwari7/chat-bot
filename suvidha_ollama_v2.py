import os 
import re
import logging
import atexit
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import speech_recognition as sr
from gtts import gTTS
import tempfile
import grpc
from google.cloud import vision_v1

# Import our Ollama integration
from ollama_integration import ollama_client

# Configure logging to suppress absl warnings
logging.basicConfig(level=logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

app = Flask(__name__)

# Initialize global variables
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Google Cloud Vision client with proper settings
def init_vision_client():
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()  # Load environment variables from .env file

        ocr_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ocr_key_path

        client_options = {'api_endpoint': 'vision.googleapis.com'}
        return vision_v1.ImageAnnotatorClient(client_options=client_options)
    except Exception as e:
        print(f"Error initializing Vision client: {e}")
        return None

# Initialize vision client
vision_client = init_vision_client()

# Initialize speech recognizer
recognizer = sr.Recognizer()

def cleanup_grpc():
    """Cleanup gRPC channels on shutdown"""
    try:
        for channel in grpc._channel._channel_pool:
            try:
                channel.close()
            except:
                continue
    except:
        pass

# Register cleanup function
atexit.register(cleanup_grpc)

def detect_text_in_image(image_path):
    """Extract text from image using Google Cloud Vision API"""
    if not vision_client:
        return "Vision client not initialized"
    
    try:
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision_v1.Image(content=content)
        response = vision_client.text_detection(
            image=image,
            timeout=30  # Set explicit timeout
        )
        
        if not response.text_annotations:
            return "No text detected"
        
        return response.text_annotations[0].description
    except Exception as e:
        print(f"Error in text detection: {e}")
        return f"Error detecting text: {str(e)}"

def get_text_chunks(text):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create vector store from text chunks using HuggingFace embeddings"""
    try:
        # Use HuggingFace embeddings for better FAISS compatibility
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        print("✅ Document vector store created successfully")
    except Exception as e:
        print(f"Error creating vector store: {e}")

def get_vector_store_image(text_chunks):
    """Create vector store for image text using HuggingFace embeddings"""
    try:
        # Use HuggingFace embeddings for better FAISS compatibility
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_image")
        print("✅ Image vector store created successfully")
    except Exception as e:
        print(f"Error creating image vector store: {e}")

def get_ollama_response(question, context=""):
    """Get response from Ollama instead of Google Gemini"""
    try:
        return ollama_client.generate_response(question, context, temperature=0.3)
    except Exception as e:
        print(f"Error getting Ollama response: {e}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"

def process_document_query(user_question):
    """Process document queries using Ollama for generation and HuggingFace for embeddings"""
    try:
        # Load FAISS vector store with HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        
        # Combine context from retrieved documents
        context = "\n".join([doc.page_content for doc in docs])
        
        # Get response from Ollama (this is the key change - using Ollama for generation)
        response = get_ollama_response(user_question, context)
        return response
    except Exception as e:
        return f"Error processing document query: {str(e)}"

def process_image_query(user_question):
    """Process image queries using Ollama for generation and HuggingFace for embeddings"""
    try:
        # Load image FAISS vector store with HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local("faiss_index_image", embeddings, allow_dangerous_deserialization=True)
        images = vector_store.similarity_search(user_question)
        
        # Combine context from retrieved image text
        context = "\n".join([img.page_content for img in images])
        
        # Get response from Ollama (this is the key change - using Ollama for generation)
        response = get_ollama_response(user_question, context)
        return response
    except Exception as e:
        return f"Error processing image query: {str(e)}"

def classify_intent(user_input):
    """Classify user intent"""
    doc_keywords = r"explain|describe|document|pdf|file|text|read|extract|analyze"
    image_keywords = r"image|picture|photo|explain|describe|text|read|extract|analyze"
    if re.search(doc_keywords, user_input, re.IGNORECASE):
        return "document"
    elif re.search(image_keywords, user_input, re.IGNORECASE):
        return "image"
    return "chat"

# Routes
@app.route('/')
def home():
    """Home page route"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat_message():
    """Handle chat messages using Ollama"""
    try:
        user_input = request.form.get('user_input')
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
            
        intent = classify_intent(user_input)
        
        if intent == "document":
            response = process_document_query(user_input)
        elif intent == "image":
            response = process_image_query(user_input)
        else:
            # General chat using Ollama
            response = get_ollama_response(user_input)
            
        return jsonify({"response": response})
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image uploads"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image = request.files['image']
    if not image.filename:
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(image.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        image.save(file_path)
        extracted_text = detect_text_in_image(file_path)
        
        if extracted_text and extracted_text != "No text detected":
            text_chunks = get_text_chunks(extracted_text)
            get_vector_store_image(text_chunks)
            return jsonify({
                "message": "Image processed successfully",
                "extracted_text": extracted_text
            })
        else:
            return jsonify({
                "message": "No text detected in image",
                "extracted_text": ""
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Handle document uploads"""
    if 'document' not in request.files:
        return jsonify({"error": "No document provided"}), 400
    
    document = request.files['document']
    if not document.filename:
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(document.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        document.save(file_path)
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
            
        if not text.strip():
            return jsonify({"error": "No text found in PDF"}), 400
            
        text_chunks = get_text_chunks(text)
        get_vector_store(text_chunks)
        return jsonify({"message": "Document processed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voice_input', methods=['POST'])
def voice_input():
    """Handle voice input"""
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            return jsonify({"text": text})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test_ollama', methods=['GET'])
def test_ollama():
    """Test endpoint to verify Ollama is working"""
    try:
        test_response = ollama_client.generate_response("What is 2+2?")
        return jsonify({
            "status": "success",
            "message": "Ollama is working correctly",
            "test_response": test_response
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Ollama test failed: {str(e)}"
        }), 500

@app.route('/test_rag', methods=['POST'])
def test_rag():
    """Test RAG functionality"""
    try:
        user_input = request.json.get('question', 'What is this document about?')
        
        # Try to process as document query
        response = process_document_query(user_input)
        
        return jsonify({
            "status": "success",
            "question": user_input,
            "response": response,
            "model": "Ollama + HuggingFace Embeddings"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"RAG test failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Initialize services before running the app
    if not vision_client:
        print("Warning: Vision client failed to initialize")
    
    # Test Ollama connection
    print("Testing Ollama connection...")
    if ollama_client.test_connection():
        print("✅ Ollama is ready!")
    else:
        print("❌ Ollama connection failed. Please check if Ollama is running.")
    
    print("🚀 Starting RAG Narok with Ollama integration...")
    print("📝 Documents will use HuggingFace embeddings for FAISS")
    print("🤖 Text generation will use Ollama (llama3.2:3b)")
    print("🌐 Server will be available at http://localhost:5000")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

