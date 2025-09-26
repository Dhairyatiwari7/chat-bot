import os 
import re
import logging
import atexit
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import docx
import speech_recognition as sr
from gtts import gTTS
import tempfile
import json

# Import our Ollama integration
from ollama_integration import ollama_client, ollama_embeddings

# Import offline audio RAG system
from audio_rag_offline import audio_rag

# Import offline image processing system
from image_processing_offline import image_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize global variables
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize speech recognizer
recognizer = sr.Recognizer()

def cleanup():
    """Cleanup on shutdown"""
    try:
        pass
    except:
        pass

# Register cleanup function
atexit.register(cleanup)

def detect_text_in_image_offline(image_path):
    """
    Extract text from image using offline Tesseract OCR
    """
    try:
        # Use the comprehensive image processor
        result = image_processor.process_image_for_rag(image_path)
        
        if result["success"]:
            return result["extracted_text"]
        else:
            return f"Error extracting text: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        print(f"Error in offline OCR: {e}")
        return f"Error extracting text from image: {str(e)}"

def extract_text_from_document(file_path):
    """Extract text from various document formats"""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            # Extract text from PDF
            pdf_reader = PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
            
        elif file_extension in ['.docx', '.doc']:
            # Extract text from Word documents
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
            
        elif file_extension in ['.txt']:
            # Extract text from plain text files
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
        else:
            return f"Unsupported file format: {file_extension}"
            
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def get_text_chunks(text):
    """Split text into chunks - optimized for speed"""
    # Smaller chunks for faster processing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,  # Reduced for faster processing
        chunk_overlap=400,  # Reduced overlap
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_text(text)

# Import LangChain base class for proper embedding interface
from langchain_core.embeddings import Embeddings

class OllamaEmbeddings(Embeddings):
    """LangChain-compatible Ollama embeddings class"""
    
    def __init__(self):
        self.client = ollama_embeddings
        
    def embed_documents(self, texts):
        """Embed multiple documents"""
        embeddings = []
        for text in texts:
            embedding = self.client.embed_text(text)
            if embedding:
                embeddings.append(embedding)
            else:
                # Fallback to zero vector if embedding fails
                embeddings.append([0.0] * 768)  # nomic-embed-text has 768 dimensions
        return embeddings
        
    def embed_query(self, text):
        """Embed a single query"""
        embedding = self.client.embed_text(text)
        if embedding:
            return embedding
        else:
            return [0.0] * 768  # Fallback to zero vector

def get_ollama_embeddings():
    """Get properly formatted Ollama embeddings for LangChain"""
    return OllamaEmbeddings()

def get_vector_store(text_chunks):
    """Create vector store from text chunks using COMPLETE Ollama embeddings"""
    try:
        # Use our proper Ollama embeddings class
        embeddings = get_ollama_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_offline")
        print("✅ Document vector store created successfully with Ollama embeddings")
    except Exception as e:
        print(f"Error creating vector store: {e}")

def get_vector_store_image(text_chunks):
    """Create vector store for image text using COMPLETE Ollama embeddings"""
    try:
        # Use our proper Ollama embeddings class
        embeddings = get_ollama_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_image_offline")
        print("✅ Image vector store created successfully with Ollama embeddings")
    except Exception as e:
        print(f"Error creating image vector store: {e}")

def get_ollama_response(question, context=""):
    """Get response from Ollama - COMPLETE OFFLINE with speed optimization"""
    try:
        # Optimize prompt for faster response
        if context:
            # Limit context length for faster processing
            context = context[:2000] if len(context) > 2000 else context
            
        return ollama_client.generate_response(question, context, temperature=0.1)  # Lower temperature for faster response
    except Exception as e:
        print(f"Error getting Ollama response: {e}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"

def process_document_query(user_question):
    """Process document queries using COMPLETE Ollama (embeddings + generation) - OPTIMIZED"""
    try:
        # Load FAISS vector store with proper Ollama embeddings
        embeddings = get_ollama_embeddings()
        vector_store = FAISS.load_local("faiss_index_offline", embeddings, allow_dangerous_deserialization=True)
        
        # Limit search results for faster processing
        docs = vector_store.similarity_search(user_question, k=3)  # Only get top 3 most relevant chunks
        
        # Combine context from retrieved documents
        context = "\n".join([doc.page_content for doc in docs])
        
        # Get response from Ollama (COMPLETE OFFLINE)
        response = get_ollama_response(user_question, context)
        return response
    except Exception as e:
        return f"Error processing document query: {str(e)}"

def process_image_query(user_question):
    """Process image queries using COMPLETE Ollama with enhanced context"""
    try:
        # Load image FAISS vector store with proper Ollama embeddings
        embeddings = get_ollama_embeddings()
        vector_store = FAISS.load_local("faiss_index_image_offline", embeddings, allow_dangerous_deserialization=True)
        
        # Get more relevant chunks for better context
        images = vector_store.similarity_search(user_question, k=5)
        
        # Combine context from retrieved image text with enhanced formatting
        context_parts = []
        for img in images:
            content = img.page_content
            if "Document Type:" in content:
                context_parts.append(f"📄 {content}")
            elif "Document Summary:" in content:
                context_parts.append(f"📋 {content}")
            elif "Document Content:" in content:
                context_parts.append(f"📝 {content}")
            elif "Detected Object:" in content:
                context_parts.append(f"🔍 {content}")
            elif "Identified Product:" in content:
                context_parts.append(f"🛍️ {content}")
            elif "Product Description:" in content:
                context_parts.append(f"📖 {content}")
            elif "Product Specifications:" in content:
                context_parts.append(f"⚙️ {content}")
            elif "Product Features:" in content:
                context_parts.append(f"✨ {content}")
            elif "Dominant Colors:" in content:
                context_parts.append(f"🎨 {content}")
            elif "Detected Shapes:" in content:
                context_parts.append(f"📐 {content}")
            else:
                context_parts.append(f"📌 {content}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt for better image-related responses
        enhanced_prompt = f"""
You are an AI assistant that analyzes images and provides detailed information about objects and products detected in them.

Based on the following extracted image content, metadata, and analysis:

{context}

Question: {user_question}

Please provide a comprehensive and accurate response based on the actual content extracted from the images. 

If the question asks about:
- What the image contains: Describe the detected objects and identified products
- Product specifications: Provide details about specifications, features, and categories
- Image analysis: Explain the visual features, colors, shapes, and characteristics
- General questions: Use the extracted text and visual analysis to answer

Be specific and reference the actual data from the image analysis. If you don't have enough information, say so clearly.
"""
        
        # Get response from Ollama with enhanced prompt
        response = get_ollama_response(enhanced_prompt, "")
        return response
    except Exception as e:
        return f"Error processing image query: {str(e)}"

def process_audio_query(user_question):
    """Process audio queries using COMPLETE Ollama (embeddings + generation)"""
    try:
        # Query audio content using the audio RAG system
        response = audio_rag.query_audio_content(user_question)
        return response
    except Exception as e:
        return f"Error processing audio query: {str(e)}"

def classify_intent(user_input):
    """Classify user intent"""
    doc_keywords = r"explain|describe|document|pdf|file|text|read|extract|analyze"
    image_keywords = r"image|picture|photo|explain|describe|text|read|extract|analyze"
    audio_keywords = r"audio|voice|speech|sound|record|transcribe|listen"
    if re.search(doc_keywords, user_input, re.IGNORECASE):
        return "document"
    elif re.search(image_keywords, user_input, re.IGNORECASE):
        return "image"
    elif re.search(audio_keywords, user_input, re.IGNORECASE):
        return "audio"
    return "chat"

# Routes
@app.route('/')
def home():
    """Home page route"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat_message():
    """Handle chat messages using COMPLETE Ollama (OFFLINE)"""
    try:
        user_input = request.form.get('user_input')
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
            
        intent = classify_intent(user_input)
        
        if intent == "document":
            response = process_document_query(user_input)
        elif intent == "image":
            response = process_image_query(user_input)
        elif intent == "audio":
            response = process_audio_query(user_input)
        else:
            # General chat using Ollama (OFFLINE)
            response = get_ollama_response(user_input)
            
        return jsonify({"response": response})
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image uploads - COMPREHENSIVE OFFLINE OCR & ANALYSIS"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image = request.files['image']
    if not image.filename:
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(image.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        image.save(file_path)
        
        # Use comprehensive offline image processing
        processing_result = image_processor.process_image_for_rag(file_path)
        
        if processing_result["success"] and processing_result["extracted_text"]:
            # Use the structured text chunks for better RAG
            text_chunks = processing_result["text_chunks"]
            get_vector_store_image(text_chunks)
            
            # Get enhanced information with safe conversion
            detected_objects = processing_result.get("detected_objects", [])
            product_identification = processing_result.get("product_identification", {})
            
            # Create response with safe data types
            response_data = {
                "message": "Image processed successfully with advanced offline OCR and object detection",
                "extracted_text": str(processing_result["extracted_text"]),
                "image_type": str(processing_result["structured_content"]["image_type"]),
                "confidence": float(processing_result["structured_content"]["confidence"]),
                "summary": str(processing_result["structured_content"]["summary"]),
                "word_count": int(processing_result["structured_content"]["word_count"]),
                "detected_objects": detected_objects,
                "product_identification": product_identification,
                "enhanced_analysis": True
            }
            
            return jsonify(response_data)
        else:
            error_msg = processing_result.get("error", "No text detected")
            return jsonify({
                "message": f"Image processing failed: {error_msg}",
                "extracted_text": "",
                "error": str(error_msg)
            })
    except Exception as e:
        logger.error(f"Image upload error: {e}")
        return jsonify({"error": f"Image processing error: {str(e)}"}), 500

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Handle multiple document uploads - OFFLINE processing"""
    if 'document' not in request.files:
        return jsonify({"error": "No document provided"}), 400
    
    files = request.files.getlist('document')
    if not files or all(not file.filename for file in files):
        return jsonify({"error": "No selected files"}), 400
    
    processed_files = []
    all_text = ""
    
    try:
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text from document
                text = extract_text_from_document(file_path)
                
                if "Error" not in text and text.strip():
                    all_text += f"\n\n=== {filename} ===\n{text}"
                    processed_files.append(filename)
                else:
                    processed_files.append(f"{filename} (failed: {text})")
        
        if not all_text.strip():
            return jsonify({"error": "No text found in any documents"}), 400
            
        # Process all documents together
        text_chunks = get_text_chunks(all_text)
        get_vector_store(text_chunks)
        
        return jsonify({
            "message": f"Successfully processed {len([f for f in processed_files if 'failed' not in f])} documents",
            "files": processed_files,
            "total_chunks": len(text_chunks)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voice_input', methods=['POST'])
def voice_input():
    """Handle voice input - OFFLINE speech recognition using Whisper"""
    temp_file_path = None
    try:
        # Check if audio file is provided
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and audio_file.filename:
                # Use a simple approach - save to uploads folder with timestamp
                import time
                timestamp = int(time.time() * 1000)  # milliseconds
                
                # Determine file extension based on content type
                content_type = audio_file.content_type or 'audio/wav'
                if 'webm' in content_type:
                    temp_filename = f"voice_{timestamp}.webm"
                else:
                    temp_filename = f"voice_{timestamp}.wav"
                
                # Use uploads folder
                temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                
                logger.info(f"Saving audio file to: {temp_file_path}")
                logger.info(f"Content type: {content_type}")
                logger.info(f"File size before save: {len(audio_file.read())}")
                audio_file.seek(0)  # Reset file pointer
                
                # Save the audio file
                audio_file.save(temp_file_path)
                
                # Verify file exists and has content
                if os.path.exists(temp_file_path):
                    file_size = os.path.getsize(temp_file_path)
                    logger.info(f"Audio file saved successfully: {temp_file_path} ({file_size} bytes)")
                    
                    if file_size > 0:
                        # Transcribe using offline Whisper
                        logger.info(f"Starting transcription of: {temp_file_path}")
                        transcribed_text = audio_rag.transcribe_audio(temp_file_path)
                        logger.info(f"Transcription result: {transcribed_text[:100]}...")
                        
                        if transcribed_text and not transcribed_text.startswith("Error"):
                            return jsonify({
                                "text": transcribed_text,
                                "method": "offline_whisper",
                                "success": True
                            })
                        else:
                            return jsonify({
                                "error": f"Transcription failed: {transcribed_text}",
                                "success": False
                            }), 400
                    else:
                        return jsonify({
                            "error": "Audio file is empty",
                            "success": False
                        }), 400
                else:
                    logger.error(f"Failed to save audio file to: {temp_file_path}")
                    return jsonify({
                        "error": "Failed to save audio file",
                        "success": False
                    }), 400
        
        return jsonify({"error": "No audio file provided", "success": False}), 400
        
    except Exception as e:
        logger.error(f"Voice input error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": f"Offline speech recognition error: {str(e)}",
            "success": False
        }), 500
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Handle audio file uploads - OFFLINE processing with Whisper"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Save uploaded audio file
        audio_file.save(file_path)
        
        # Process audio file with offline Whisper
        result = audio_rag.process_audio_file(file_path)
        
        if result["success"]:
            return jsonify({
                "message": "Audio processed successfully with offline Whisper",
                "transcribed_text": result["transcribed_text"],
                "chunks_created": result["chunks_created"],
                "vector_store_created": result["vector_store_created"]
            })
        else:
            return jsonify({"error": result["message"]}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test_ollama', methods=['GET'])
def test_ollama():
    """Test endpoint to verify Ollama is working"""
    try:
        test_response = ollama_client.generate_response("What is 2+2?")
        return jsonify({
            "status": "success",
            "message": "Ollama is working correctly (COMPLETE OFFLINE)",
            "test_response": test_response,
            "model": "llama3.2:3b",
            "embeddings": "nomic-embed-text"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Ollama test failed: {str(e)}"
        }), 500

@app.route('/test_voice', methods=['GET'])
def test_voice():
    """Test endpoint to verify voice input system is working"""
    try:
        # Test Whisper model
        whisper_status = "✅ Loaded" if audio_rag.whisper_model else "❌ Not loaded"
        
        # Test Ollama connection
        ollama_status = ollama_client.test_connection()
        
        return jsonify({
            "status": "success",
            "message": "Voice input system status",
            "whisper_model": whisper_status,
            "ollama_connection": "✅ Connected" if ollama_status else "❌ Disconnected",
            "supported_formats": ["wav", "webm", "mp3", "m4a", "flac", "ogg"],
            "features": {
                "real_time_recording": "✅ Available",
                "offline_transcription": "✅ Whisper",
                "offline_generation": "✅ Ollama",
                "auto_send": "✅ Enabled"
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Voice test failed: {str(e)}"
        }), 500

@app.route('/test_rag_offline', methods=['POST'])
def test_rag_offline():
    """Test RAG functionality - COMPLETE OFFLINE"""
    try:
        user_input = request.json.get('question', 'What is this document about?')
        
        # Try to process as document query
        response = process_document_query(user_input)
        
        return jsonify({
            "status": "success",
            "question": user_input,
            "response": response,
            "model": "Ollama llama3.2:3b",
            "embeddings": "Ollama nomic-embed-text",
            "mode": "COMPLETE OFFLINE"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"RAG test failed: {str(e)}"
        }), 500

@app.route('/audio_status', methods=['GET'])
def audio_status():
    """Get audio system status"""
    try:
        audio_status_info = audio_rag.get_audio_status()
        return jsonify({
            "system": "RAG Narok - Audio RAG System",
            "whisper_model": audio_status_info["whisper_model"],
            "audio_vector_store": audio_status_info["audio_vector_store"],
            "supported_formats": audio_status_info["supported_formats"],
            "model_size": audio_status_info["model_size"]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Audio status check failed: {str(e)}"
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Get system status"""
    try:
        ollama_status = ollama_client.test_connection()
        audio_status_info = audio_rag.get_audio_status()
        
        return jsonify({
            "system": "RAG Narok - Complete Offline Version",
            "ollama_status": "✅ Connected" if ollama_status else "❌ Disconnected",
            "models": {
                "llm": "llama3.2:3b",
                "embeddings": "nomic-embed-text",
                "speech_to_text": "Whisper base (offline)"
            },
            "features": {
                "text_generation": "✅ Offline (Ollama)",
                "embeddings": "✅ Offline (Ollama)",
                "pdf_processing": "✅ Offline (PyPDF2)",
                "image_ocr": "✅ Offline (Tesseract + OpenCV)",
                "image_analysis": "✅ Advanced (Object Detection + Product ID)",
                "voice_input": "✅ Real-time (Whisper + Auto-send)",
                "audio_rag": "✅ Offline (Whisper + Ollama)",
                "real_time_speech": "✅ Enhanced (WebM/WAV + Noise Reduction)"
            },
            "storage": {
                "documents": "faiss_index_offline/",
                "images": "faiss_index_image_offline/",
                "audio": "faiss_index_audio_offline/"
            },
            "audio_system": audio_status_info
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Status check failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Test Ollama connection
    print("🚀 Testing COMPLETE OFFLINE Ollama setup...")
    print("=" * 50)
    
    if ollama_client.test_connection():
        print("✅ Ollama LLM (llama3.2:3b) is ready!")
    else:
        print("❌ Ollama LLM connection failed. Please check if Ollama is running.")
        exit(1)
    
    # Test embeddings
    test_embedding = ollama_embeddings.embed_text("test")
    if test_embedding:
        print(f"✅ Ollama Embeddings (nomic-embed-text) is ready! Dimension: {len(test_embedding)}")
    else:
        print("❌ Ollama Embeddings failed.")
        exit(1)
    
    # Test audio system
    print("🎤 Testing Audio RAG System...")
    if audio_rag.whisper_model:
        print("✅ Whisper model loaded successfully")
    else:
        print("❌ Whisper model failed to load")
    
    print("=" * 50)
    print("🎉 RAG Narok - COMPLETE OFFLINE VERSION")
    print("🤖 Text Generation: Ollama llama3.2:3b")
    print("🔤 Embeddings: Ollama nomic-embed-text")
    print("📄 PDF Processing: PyPDF2 (offline)")
    print("🖼️ Image OCR: Tesseract + OpenCV (COMPLETE OFFLINE)")
    print("🎤 Voice: Whisper (COMPLETE OFFLINE)")
    print("🎵 Audio RAG: Whisper + Ollama (COMPLETE OFFLINE)")
    print("🌐 Server: http://localhost:5000")
    print("📊 Status: http://localhost:5000/status")
    print("🎵 Audio Status: http://localhost:5000/audio_status")
    print("=" * 50)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
