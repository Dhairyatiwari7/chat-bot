"""
Offline Audio RAG System using Whisper and Ollama
This module handles offline speech-to-text and audio-based RAG functionality
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Optional
import whisper
from ollama_integration import ollama_client, ollama_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioRAGSystem:
    """Offline Audio RAG System using Whisper and Ollama"""
    
    def __init__(self):
        """Initialize the audio RAG system"""
        self.whisper_model = None
        self.audio_vector_store = None
        self.load_whisper_model()
        
    def load_whisper_model(self):
        """Load Whisper model for offline speech recognition"""
        try:
            # Use 'base' model for good balance of speed and accuracy
            # Options: tiny, base, small, medium, large
            self.whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            self.whisper_model = None
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribe audio file to text using offline Whisper
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            if not self.whisper_model:
                return "Whisper model not loaded"
            
            # Convert to absolute path and normalize for Windows
            audio_file_path = os.path.abspath(os.path.normpath(audio_file_path))
            
            # Check if file exists and has content
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Temp directory: {tempfile.gettempdir()}")
                logger.error(f"Upload folder: {os.path.abspath('uploads')}")
                return "Audio file not found"
            
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                logger.error(f"Audio file is empty: {audio_file_path}")
                return "Audio file is empty"
            
            logger.info(f"Transcribing audio file: {audio_file_path} ({file_size} bytes)")
            logger.info(f"File exists: {os.path.exists(audio_file_path)}")
            logger.info(f"File readable: {os.access(audio_file_path, os.R_OK)}")
            
            # Add a small delay to ensure file is fully written
            import time
            time.sleep(0.2)
            
            # Double-check file still exists after delay
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file disappeared: {audio_file_path}")
                return "Audio file disappeared"
            
            # Try to convert WebM to WAV if needed (simple approach)
            processed_file_path = audio_file_path
            if audio_file_path.lower().endswith('.webm'):
                try:
                    # For now, try to transcribe WebM directly
                    # Whisper should handle WebM format
                    logger.info("Attempting to transcribe WebM file directly")
                except Exception as conv_error:
                    logger.warning(f"WebM conversion issue: {conv_error}")
            
            # Try to load audio using librosa first (doesn't require ffmpeg)
            try:
                import librosa
                import soundfile as sf
                
                # Load audio with librosa
                audio_data, sample_rate = librosa.load(processed_file_path, sr=16000)
                logger.info(f"Audio loaded with librosa: {len(audio_data)} samples at {sample_rate}Hz")
                
                # Transcribe using the audio data directly
                result = self.whisper_model.transcribe(
                    audio_data,
                    fp16=False,  # Use FP32 for better compatibility
                    language="en",  # Set language for better accuracy
                    task="transcribe",
                    verbose=False  # Reduce logging noise
                )
                
            except ImportError:
                logger.warning("librosa not available, trying direct file transcription")
                # Fallback to direct file transcription
                result = self.whisper_model.transcribe(
                    processed_file_path,
                    fp16=False,  # Use FP32 for better compatibility
                    language="en",  # Set language for better accuracy
                    task="transcribe",
                    verbose=False  # Reduce logging noise
                )
            except Exception as librosa_error:
                logger.warning(f"librosa failed: {librosa_error}, trying direct file transcription")
                # Fallback to direct file transcription
                result = self.whisper_model.transcribe(
                    processed_file_path,
                    fp16=False,  # Use FP32 for better compatibility
                    language="en",  # Set language for better accuracy
                    task="transcribe",
                    verbose=False  # Reduce logging noise
                )
            
            transcribed_text = result["text"].strip()
            
            if transcribed_text:
                logger.info(f"Audio transcribed successfully: {len(transcribed_text)} characters")
                return transcribed_text
            else:
                logger.warning("No speech detected in audio")
                return "No speech detected in audio"
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return f"Error transcribing audio: {str(e)}"
    
    def process_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Process audio file and create vector store
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Transcribe audio
            transcribed_text = self.transcribe_audio(audio_file_path)
            
            if not transcribed_text or "Error" in transcribed_text:
                return {
                    "success": False,
                    "message": "Failed to transcribe audio",
                    "transcribed_text": transcribed_text
                }
            
            # Create text chunks
            text_chunks = self.create_text_chunks(transcribed_text)
            
            # Create vector store
            vector_store_created = self.create_audio_vector_store(text_chunks)
            
            return {
                "success": True,
                "message": "Audio processed successfully",
                "transcribed_text": transcribed_text,
                "chunks_created": len(text_chunks),
                "vector_store_created": vector_store_created
            }
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return {
                "success": False,
                "message": f"Error processing audio: {str(e)}",
                "transcribed_text": ""
            }
    
    def create_text_chunks(self, text: str) -> List[str]:
        """Create text chunks from transcribed audio"""
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            chunks = splitter.split_text(text)
            return chunks
        except Exception as e:
            logger.error(f"Error creating text chunks: {e}")
            return [text]  # Return original text as single chunk
    
    def create_audio_vector_store(self, text_chunks: List[str]) -> bool:
        """Create vector store for audio text using Ollama embeddings"""
        try:
            # Use our proper Ollama embeddings class
            embeddings = self.get_ollama_embeddings()
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index_audio_offline")
            logger.info("✅ Audio vector store created successfully with Ollama embeddings")
            return True
        except Exception as e:
            logger.error(f"Error creating audio vector store: {e}")
            return False
    
    def get_ollama_embeddings(self):
        """Get properly formatted Ollama embeddings for LangChain"""
        class OllamaEmbeddings(Embeddings):
            def __init__(self):
                self.client = ollama_embeddings
                
            def embed_documents(self, texts):
                embeddings = []
                for text in texts:
                    embedding = self.client.embed_text(text)
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        embeddings.append([0.0] * 768)
                return embeddings
                
            def embed_query(self, text):
                embedding = self.client.embed_text(text)
                if embedding:
                    return embedding
                else:
                    return [0.0] * 768
        
        return OllamaEmbeddings()
    
    def query_audio_content(self, question: str) -> str:
        """
        Query audio content using RAG
        
        Args:
            question: User's question about the audio content
            
        Returns:
            Answer based on audio content
        """
        try:
            # Load audio vector store
            embeddings = self.get_ollama_embeddings()
            vector_store = FAISS.load_local("faiss_index_audio_offline", embeddings, allow_dangerous_deserialization=True)
            
            # Search for relevant content
            docs = vector_store.similarity_search(question)
            
            # Combine context from retrieved audio text
            context = "\n".join([doc.page_content for doc in docs])
            
            # Get response from Ollama
            response = ollama_client.generate_response(question, context, temperature=0.3)
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying audio content: {e}")
            return f"Error processing audio query: {str(e)}"
    
    def get_audio_status(self) -> Dict[str, Any]:
        """Get status of audio RAG system"""
        try:
            whisper_status = "✅ Loaded" if self.whisper_model else "❌ Not loaded"
            
            # Check if audio vector store exists
            audio_store_exists = os.path.exists("faiss_index_audio_offline/index.faiss")
            vector_store_status = "✅ Available" if audio_store_exists else "⚠️ No audio content indexed"
            
            return {
                "whisper_model": whisper_status,
                "audio_vector_store": vector_store_status,
                "supported_formats": ["wav", "mp3", "m4a", "flac", "ogg"],
                "model_size": "base (39 MB)"
            }
            
        except Exception as e:
            logger.error(f"Error getting audio status: {e}")
            return {"error": str(e)}

# Global audio RAG instance
audio_rag = AudioRAGSystem()

def test_audio_system():
    """Test the audio RAG system"""
    print("🎤 Testing Offline Audio RAG System...")
    print("=" * 50)
    
    # Test Whisper loading
    if audio_rag.whisper_model:
        print("✅ Whisper model loaded successfully")
    else:
        print("❌ Whisper model failed to load")
        return False
    
    # Test Ollama connection
    if ollama_client.test_connection():
        print("✅ Ollama connection working")
    else:
        print("❌ Ollama connection failed")
        return False
    
    # Get status
    status = audio_rag.get_audio_status()
    print(f"📊 Audio Status: {status}")
    
    print("=" * 50)
    print("🎉 Audio RAG System Ready!")
    print("🎤 Speech-to-Text: Whisper (offline)")
    print("🤖 Text Generation: Ollama (offline)")
    print("🔤 Embeddings: Ollama (offline)")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_audio_system()
