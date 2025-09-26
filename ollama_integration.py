"""
Ollama Integration for RAG Narok
This module handles all Ollama-related functionality
"""

import ollama
import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient:
    """Ollama client for handling LLM operations"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize Ollama client
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.client = ollama.Client()
        
    def generate_response(self, prompt: str, context: str = "", temperature: float = 0.3) -> str:
        """
        Generate response using Ollama model
        
        Args:
            prompt: The user's question
            context: Additional context from RAG
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            Generated response
        """
        try:
            # Create the full prompt with context
            full_prompt = self._create_prompt(prompt, context)
            
            # Generate response
            response = self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={
                    'temperature': temperature,
                    'top_p': 0.9,
                    'max_tokens': 1000
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create formatted prompt for the model
        
        Args:
            question: User's question
            context: Retrieved context from RAG
            
        Returns:
            Formatted prompt
        """
        if context:
            prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context. 
If the answer is not in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""
        else:
            prompt = f"""You are a helpful AI assistant. Please answer the following question:

Question: {question}

Answer:"""
        
        return prompt
    
    def chat_with_history(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat with conversation history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Assistant's response
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9
                }
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I apologize, but I encountered an error while processing your request."
    
    def test_connection(self) -> bool:
        """
        Test if Ollama is running and accessible
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt="Hello, are you working?",
                options={'max_tokens': 10}
            )
            return True
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False

class OllamaEmbeddings:
    """Ollama embeddings for vector operations"""
    
    def __init__(self, model_name: str = "nomic-embed-text"):
        """
        Initialize Ollama embeddings
        
        Args:
            model_name: Name of the embedding model
        """
        self.model_name = model_name
        self.client = ollama.Client()
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for doc in documents:
            embedding = self.embed_text(doc)
            if embedding:
                embeddings.append(embedding)
            else:
                logger.warning(f"Failed to generate embedding for document: {doc[:100]}...")
        return embeddings

# Global instances
ollama_client = OllamaClient()
ollama_embeddings = OllamaEmbeddings()

def test_ollama_setup():
    """Test if Ollama is properly set up"""
    print("Testing Ollama setup...")
    
    # Test client connection
    if ollama_client.test_connection():
        print("✅ Ollama client connection successful")
    else:
        print("❌ Ollama client connection failed")
        return False
    
    # Test embeddings
    test_embedding = ollama_embeddings.embed_text("test")
    if test_embedding:
        print("✅ Ollama embeddings working")
        print(f"Embedding dimension: {len(test_embedding)}")
    else:
        print("❌ Ollama embeddings failed")
        return False
    
    # Test generation
    test_response = ollama_client.generate_response("What is 2+2?")
    if test_response and "4" in test_response:
        print("✅ Ollama text generation working")
        print(f"Sample response: {test_response}")
    else:
        print("❌ Ollama text generation failed")
        return False
    
    print("🎉 All Ollama tests passed!")
    return True

if __name__ == "__main__":
    test_ollama_setup()
