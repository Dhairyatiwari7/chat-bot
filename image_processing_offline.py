"""
Offline Image Processing Module for RAG Narok
Handles OCR, image analysis, and text extraction using Tesseract and OpenCV
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import logging
from typing import Dict, List, Tuple, Optional
import base64
import io
import json
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        """Initialize the offline image processor"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        self.tesseract_config = r'--oem 3 --psm 6'  # Optimized for document text
        
        # Test Tesseract availability
        try:
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR is available")
            self.tesseract_available = True
        except Exception as e:
            logger.error(f"❌ Tesseract OCR not found: {e}")
            logger.info("Please install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            self.tesseract_available = False
        
        # Initialize object detection
        self.object_detector = None
        self.init_object_detection()
        
        # Product knowledge base
        self.product_knowledge = self.load_product_knowledge()
    
    def init_object_detection(self):
        """Initialize offline object detection using OpenCV DNN"""
        try:
            # Try to load a pre-trained model (YOLO or similar)
            # For now, we'll use OpenCV's built-in Haar cascades and HOG detector
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Load Haar cascades for common objects
            cascade_paths = {
                'car': cv2.data.haarcascades + 'haarcascade_car.xml',
                'face': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                'eye': cv2.data.haarcascades + 'haarcascade_eye.xml'
            }
            
            self.cascades = {}
            for name, path in cascade_paths.items():
                if os.path.exists(path):
                    self.cascades[name] = cv2.CascadeClassifier(path)
            
            logger.info("✅ Object detection initialized with OpenCV")
            self.object_detection_available = True
            
        except Exception as e:
            logger.warning(f"⚠️ Object detection not available: {e}")
            self.object_detection_available = False
    
    def load_product_knowledge(self) -> Dict:
        """Load product knowledge base for intelligent responses"""
        return {
            "car": {
                "description": "A motor vehicle with four wheels, typically powered by an internal combustion engine or electric motor",
                "specifications": ["Engine type", "Fuel efficiency", "Horsepower", "Transmission", "Safety features", "Seating capacity"],
                "common_features": ["Air conditioning", "Power steering", "ABS brakes", "Airbags", "GPS navigation"],
                "categories": ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Truck", "Electric Vehicle"]
            },
            "phone": {
                "description": "A mobile communication device that combines telephone and computer functions",
                "specifications": ["Screen size", "Processor", "RAM", "Storage", "Camera", "Battery life", "Operating system"],
                "common_features": ["Touchscreen", "Camera", "GPS", "WiFi", "Bluetooth", "App store", "Fingerprint scanner"],
                "categories": ["Smartphone", "Feature phone", "Gaming phone", "Business phone", "Budget phone"]
            },
            "bat": {
                "description": "A sports equipment used in games like cricket or baseball to hit the ball",
                "specifications": ["Length", "Weight", "Material", "Handle type", "Sweet spot size"],
                "common_features": ["Grip tape", "Weight distribution", "Balance point", "Durability"],
                "categories": ["Cricket bat", "Baseball bat", "Softball bat", "Training bat"]
            },
            "laptop": {
                "description": "A portable personal computer with a screen and keyboard",
                "specifications": ["Screen size", "Processor", "RAM", "Storage", "Graphics card", "Battery life", "Weight"],
                "common_features": ["Keyboard", "Touchpad", "Webcam", "Speakers", "USB ports", "HDMI port"],
                "categories": ["Gaming laptop", "Business laptop", "Ultrabook", "2-in-1 laptop", "Budget laptop"]
            },
            "book": {
                "description": "A written or printed work consisting of pages bound together",
                "specifications": ["Page count", "Dimensions", "Binding type", "Paper quality", "Font size"],
                "common_features": ["Cover design", "Table of contents", "Index", "Illustrations", "Glossary"],
                "categories": ["Fiction", "Non-fiction", "Textbook", "Reference book", "E-book"]
            },
            "chair": {
                "description": "A piece of furniture for sitting on, typically having a back and four legs",
                "specifications": ["Height", "Width", "Depth", "Material", "Weight capacity", "Adjustability"],
                "common_features": ["Armrests", "Cushioning", "Swivel base", "Height adjustment", "Lumbar support"],
                "categories": ["Office chair", "Dining chair", "Recliner", "Folding chair", "Gaming chair"]
            }
        }
    
    def detect_objects_opencv(self, image_path: str) -> List[Dict]:
        """Detect objects using OpenCV methods"""
        if not self.object_detection_available:
            return []
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            detected_objects = []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect people
            try:
                (rects, weights) = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                for (x, y, w, h) in rects:
                    detected_objects.append({
                        "class": "person",
                        "confidence": 0.8,
                        "bbox": [x, y, w, h],
                        "description": "A person detected in the image"
                    })
            except:
                pass
            
            # Detect faces
            if 'face' in self.cascades:
                faces = self.cascades['face'].detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    detected_objects.append({
                        "class": "face",
                        "confidence": 0.7,
                        "bbox": [x, y, w, h],
                        "description": "A human face detected in the image"
                    })
            
            # Detect cars (if cascade available)
            if 'car' in self.cascades:
                cars = self.cascades['car'].detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in cars:
                    detected_objects.append({
                        "class": "car",
                        "confidence": 0.6,
                        "bbox": [x, y, w, h],
                        "description": "A vehicle/car detected in the image"
                    })
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
    
    def analyze_image_features(self, image_path: str) -> Dict:
        """Analyze image features to identify objects/products"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {}
            
            height, width = image.shape[:2]
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Analyze color distribution
            dominant_colors = self.get_dominant_colors(image)
            
            # Analyze shapes and contours
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze texture
            texture_features = self.analyze_texture(gray)
            
            # Detect geometric shapes
            shapes = self.detect_shapes(contours)
            
            return {
                "dominant_colors": dominant_colors,
                "texture_features": texture_features,
                "shapes": shapes,
                "contour_count": int(len(contours)),
                "aspect_ratio": float(width / height),
                "brightness": float(np.mean(gray)),
                "contrast": float(np.std(gray))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image features: {e}")
            return {}
    
    def get_dominant_colors(self, image, k=5):
        """Get dominant colors in the image"""
        try:
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
            
            # Use K-means to find dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get the dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Count occurrences of each color
            color_counts = np.bincount(labels)
            
            # Create color names
            color_names = []
            for color in colors:
                color_names.append(self.rgb_to_color_name(color))
            
            return {
                "colors": colors.tolist(),
                "names": color_names,
                "counts": color_counts.tolist()
            }
        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            return {"colors": [], "names": [], "counts": []}
    
    def rgb_to_color_name(self, rgb):
        """Convert RGB values to color names"""
        r, g, b = rgb
        
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r > 150 and g < 100 and b > 150:
            return "purple"
        elif r > 100 and g > 100 and b > 100:
            return "gray"
        else:
            return "mixed"
    
    def analyze_texture(self, gray_image):
        """Analyze texture features of the image"""
        try:
            # Calculate Local Binary Pattern (simplified)
            # This is a basic texture analysis
            texture_features = {
                "smoothness": float(np.std(cv2.Laplacian(gray_image, cv2.CV_64F))),
                "uniformity": float(np.var(gray_image)),
                "contrast": float(np.std(gray_image))
            }
            return texture_features
        except Exception as e:
            logger.warning(f"Texture analysis failed: {e}")
            return {"smoothness": 0.0, "uniformity": 0.0, "contrast": 0.0}
    
    def detect_shapes(self, contours):
        """Detect basic shapes in contours"""
        shapes = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                # Approximate the contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Determine shape based on number of vertices
                vertices = len(approx)
                if vertices == 3:
                    shapes.append("triangle")
                elif vertices == 4:
                    shapes.append("rectangle")
                elif vertices > 8:
                    shapes.append("circle")
                else:
                    shapes.append("polygon")
        
        return list(set(shapes))  # Remove duplicates
    
    def identify_product_from_features(self, features: Dict, detected_objects: List[Dict]) -> Dict:
        """Identify products based on image features and detected objects"""
        product_info = {
            "identified_products": [],
            "confidence": 0,
            "reasoning": []
        }
        
        # Check detected objects first
        for obj in detected_objects:
            if obj["class"] in self.product_knowledge:
                product_info["identified_products"].append({
                    "product": obj["class"],
                    "confidence": obj["confidence"],
                    "knowledge": self.product_knowledge[obj["class"]],
                    "source": "object_detection"
                })
                product_info["reasoning"].append(f"Detected {obj['class']} using computer vision")
        
        # Analyze features for product identification
        dominant_colors = features.get("dominant_colors", {})
        shapes = features.get("shapes", [])
        aspect_ratio = features.get("aspect_ratio", 1)
        
        # Enhanced heuristic-based product identification
        if "rectangle" in shapes and aspect_ratio > 1.5:
            product_info["identified_products"].append({
                "product": "phone",
                "confidence": 0.6,
                "knowledge": self.product_knowledge["phone"],
                "source": "shape_analysis",
                "reasoning": "Rectangular shape with high aspect ratio suggests mobile phone"
            })
        
        if "rectangle" in shapes and aspect_ratio < 1.2:
            product_info["identified_products"].append({
                "product": "book",
                "confidence": 0.5,
                "knowledge": self.product_knowledge["book"],
                "source": "shape_analysis",
                "reasoning": "Rectangular shape with low aspect ratio suggests book"
            })
        
        # Additional heuristics based on colors and shapes
        color_names = dominant_colors.get("names", [])
        
        # If image has metallic colors and rectangular shapes, might be a laptop
        if any(color in ["gray", "white", "black"] for color in color_names) and "rectangle" in shapes:
            product_info["identified_products"].append({
                "product": "laptop",
                "confidence": 0.4,
                "knowledge": self.product_knowledge["laptop"],
                "source": "color_shape_analysis",
                "reasoning": "Metallic colors with rectangular shape suggests laptop"
            })
        
        # If image has brown colors and rectangular shapes, might be furniture
        if any(color in ["brown", "gray"] for color in color_names) and "rectangle" in shapes:
            product_info["identified_products"].append({
                "product": "chair",
                "confidence": 0.3,
                "knowledge": self.product_knowledge["chair"],
                "source": "color_shape_analysis",
                "reasoning": "Brown/gray colors with rectangular shape suggests furniture"
            })
        
        # If image has bright colors and circular shapes, might be sports equipment
        if any(color in ["red", "blue", "green", "yellow"] for color in color_names) and "circle" in shapes:
            product_info["identified_products"].append({
                "product": "bat",
                "confidence": 0.3,
                "knowledge": self.product_knowledge["bat"],
                "source": "color_shape_analysis",
                "reasoning": "Bright colors with circular elements suggests sports equipment"
            })
        
        # Calculate overall confidence
        if product_info["identified_products"]:
            confidences = [p["confidence"] for p in product_info["identified_products"]]
            product_info["confidence"] = max(confidences)
        
        return product_info
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        Preprocess image for better OCR results
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image quality
            # Increase contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Increase sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Convert to grayscale for better OCR
            gray_image = image.convert('L')
            
            # Apply noise reduction
            denoised = gray_image.filter(ImageFilter.MedianFilter(size=3))
            
            return denoised
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return Image.open(image_path)
    
    def extract_text_with_fallback(self, image_path: str) -> Dict[str, any]:
        """
        Fallback text extraction when Tesseract is not available
        """
        try:
            from PIL import Image
            import os
            
            # Get basic image information
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format or "Unknown"
            
            filename = os.path.basename(image_path)
            
            return {
                "extracted_text": f"Image Analysis: {filename} ({width}x{height} {mode} {format_name} format). OCR not available - please install Tesseract for text extraction.",
                "confidence_scores": [],
                "average_confidence": 0.0,
                "alternative_texts": [],
                "word_count": 0,
                "method": "fallback_analysis",
                "image_info": {
                    "filename": filename,
                    "dimensions": f"{width}x{height}",
                    "mode": mode,
                    "format": format_name
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback OCR error: {e}")
            return {
                "extracted_text": f"Error analyzing image: {str(e)}",
                "confidence_scores": [],
                "average_confidence": 0.0,
                "alternative_texts": [],
                "word_count": 0,
                "method": "fallback_error",
                "error": str(e)
            }

    def extract_text_with_ocr(self, image_path: str) -> Dict[str, any]:
        """
        Extract text from image using Tesseract OCR with multiple methods
        """
        # Use fallback if Tesseract is not available
        if not self.tesseract_available:
            return self.extract_text_with_fallback(image_path)
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Method 1: Standard OCR
            standard_text = pytesseract.image_to_string(
                processed_image, 
                config=self.tesseract_config
            ).strip()
            
            # Method 2: OCR with confidence scores
            data = pytesseract.image_to_data(
                processed_image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter text with confidence > 30
            confident_text = []
            confidences = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:
                    text = data['text'][i].strip()
                    if text:
                        confident_text.append(text)
                        confidences.append(int(data['conf'][i]))
            
            # Method 3: Try different PSM modes for better results
            alternative_texts = []
            for psm in [3, 4, 6, 8, 13]:
                try:
                    alt_text = pytesseract.image_to_string(
                        processed_image,
                        config=f'--oem 3 --psm {psm}'
                    ).strip()
                    if alt_text and len(alt_text) > len(standard_text) * 0.5:
                        alternative_texts.append(alt_text)
                except:
                    continue
            
            # Combine all methods
            all_text = [standard_text] + alternative_texts
            all_text = [text for text in all_text if text.strip()]
            
            # Get the longest and most complete text
            best_text = max(all_text, key=len) if all_text else standard_text
            
            return {
                "extracted_text": best_text,
                "confidence_scores": [int(c) for c in confidences],
                "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "alternative_texts": alternative_texts,
                "word_count": int(len(best_text.split())),
                "method": "tesseract_ocr"
            }
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}")
            return {
                "extracted_text": f"Error extracting text: {str(e)}",
                "confidence_scores": [],
                "average_confidence": 0.0,
                "alternative_texts": [],
                "word_count": 0,
                "method": "error",
                "error": str(e)
            }
    
    def analyze_image_content(self, image_path: str) -> Dict[str, any]:
        """
        Analyze image content and provide detailed information
        """
        try:
            # Extract text using OCR
            ocr_result = self.extract_text_with_ocr(image_path)
            
            # Basic image analysis
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            height, width = image.shape[:2]
            
            # Analyze image characteristics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect edges (indicates text regions)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # Detect contours (potential text blocks)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze brightness and contrast
            mean_brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Determine image type
            image_type = self.classify_image_type(ocr_result["extracted_text"], edge_density, mean_brightness)
            
            return {
                "ocr_result": ocr_result,
                "image_analysis": {
                    "dimensions": {"width": int(width), "height": int(height)},
                    "edge_density": float(edge_density),
                    "mean_brightness": float(mean_brightness),
                    "contrast": float(contrast),
                    "contour_count": int(len(contours)),
                    "image_type": image_type
                },
                "summary": self.generate_image_summary(ocr_result, image_type)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": f"Image analysis failed: {str(e)}"}
    
    def classify_image_type(self, text: str, edge_density: float, brightness: float) -> str:
        """
        Classify the type of image based on content
        """
        text_lower = text.lower()
        
        # Document indicators
        if any(word in text_lower for word in ['invoice', 'receipt', 'bill', 'statement', 'payment']):
            return "financial_document"
        elif any(word in text_lower for word in ['contract', 'agreement', 'terms', 'conditions']):
            return "legal_document"
        elif any(word in text_lower for word in ['report', 'analysis', 'data', 'chart', 'graph']):
            return "report"
        elif any(word in text_lower for word in ['email', 'message', 'correspondence']):
            return "correspondence"
        elif any(word in text_lower for word in ['form', 'application', 'registration']):
            return "form"
        elif edge_density > 0.1 and brightness < 150:
            return "text_document"
        elif len(text.split()) > 50:
            return "document"
        elif len(text.split()) > 10:
            return "text_image"
        else:
            return "image"
    
    def generate_image_summary(self, ocr_result: Dict, image_type: str) -> str:
        """
        Generate a summary of the image content
        """
        text = ocr_result["extracted_text"]
        word_count = ocr_result["word_count"]
        confidence = ocr_result["average_confidence"]
        
        if word_count == 0:
            return "No text detected in the image."
        
        summary_parts = []
        
        # Add type information
        summary_parts.append(f"Image type: {image_type.replace('_', ' ').title()}")
        
        # Add text statistics
        summary_parts.append(f"Text extracted: {word_count} words")
        summary_parts.append(f"OCR confidence: {confidence:.1f}%")
        
        # Add key information based on image type
        if image_type == "financial_document":
            if any(word in text.lower() for word in ['total', 'amount', 'sum', '$', '₹', '€', '£']):
                summary_parts.append("Contains financial information")
        elif image_type == "legal_document":
            if any(word in text.lower() for word in ['party', 'agreement', 'signature', 'date']):
                summary_parts.append("Contains legal/contractual information")
        
        # Add first few words as preview
        words = text.split()[:10]
        if words:
            preview = " ".join(words) + "..." if len(words) == 10 else " ".join(words)
            summary_parts.append(f"Preview: {preview}")
        
        return " | ".join(summary_parts)
    
    def process_image_for_rag(self, image_path: str) -> Dict[str, any]:
        """
        Complete image processing for RAG system with object detection and product identification
        """
        try:
            # Analyze image content (OCR)
            analysis = self.analyze_image_content(image_path)
            
            if "error" in analysis:
                return analysis
            
            # Detect objects using computer vision
            detected_objects = self.detect_objects_opencv(image_path)
            
            # Analyze image features
            image_features = self.analyze_image_features(image_path)
            
            # Identify products based on features and detected objects
            product_identification = self.identify_product_from_features(image_features, detected_objects)
            
            # Extract text for RAG
            extracted_text = analysis["ocr_result"]["extracted_text"]
            
            # Create enhanced structured content for RAG
            structured_content = {
                "raw_text": extracted_text,
                "summary": analysis["summary"],
                "image_type": analysis["image_analysis"]["image_type"],
                "confidence": float(analysis["ocr_result"]["average_confidence"]),
                "word_count": int(analysis["ocr_result"]["word_count"]),
                "metadata": self.convert_numpy_types(analysis["image_analysis"]),
                "detected_objects": self.convert_numpy_types(detected_objects),
                "image_features": self.convert_numpy_types(image_features),
                "product_identification": self.convert_numpy_types(product_identification)
            }
            
            # Create enhanced searchable text chunks
            text_chunks = self.create_enhanced_text_chunks(structured_content)
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "structured_content": structured_content,
                "text_chunks": text_chunks,
                "analysis": self.convert_numpy_types(analysis),
                "detected_objects": self.convert_numpy_types(detected_objects),
                "product_identification": self.convert_numpy_types(product_identification)
            }
            
        except Exception as e:
            logger.error(f"Error processing image for RAG: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "text_chunks": []
            }
    
    def create_text_chunks(self, structured_content: Dict) -> List[str]:
        """
        Create text chunks for RAG indexing
        """
        chunks = []
        
        # Add main text content
        if structured_content["raw_text"]:
            chunks.append(f"Document Content: {structured_content['raw_text']}")
        
        # Add metadata as searchable content
        chunks.append(f"Document Type: {structured_content['image_type']}")
        chunks.append(f"Document Summary: {structured_content['summary']}")
        
        # Add confidence information
        if structured_content["confidence"] > 0:
            chunks.append(f"Text Extraction Confidence: {structured_content['confidence']:.1f}%")
        
        return chunks
    
    def create_enhanced_text_chunks(self, structured_content: Dict) -> List[str]:
        """
        Create enhanced text chunks for RAG indexing with object detection and product identification
        """
        chunks = []
        
        # Add main text content
        if structured_content["raw_text"]:
            chunks.append(f"Document Content: {structured_content['raw_text']}")
        
        # Add metadata as searchable content
        chunks.append(f"Document Type: {structured_content['image_type']}")
        chunks.append(f"Document Summary: {structured_content['summary']}")
        
        # Add confidence information
        if structured_content["confidence"] > 0:
            chunks.append(f"Text Extraction Confidence: {structured_content['confidence']:.1f}%")
        
        # Add detected objects information
        detected_objects = structured_content.get("detected_objects", [])
        if detected_objects:
            for obj in detected_objects:
                chunks.append(f"Detected Object: {obj['class']} - {obj['description']} (Confidence: {obj['confidence']:.2f})")
        
        # Add product identification information
        product_identification = structured_content.get("product_identification", {})
        if product_identification.get("identified_products"):
            for product in product_identification["identified_products"]:
                product_name = product["product"]
                knowledge = product["knowledge"]
                chunks.append(f"Identified Product: {product_name}")
                chunks.append(f"Product Description: {knowledge['description']}")
                chunks.append(f"Product Specifications: {', '.join(knowledge['specifications'])}")
                chunks.append(f"Product Features: {', '.join(knowledge['common_features'])}")
                chunks.append(f"Product Categories: {', '.join(knowledge['categories'])}")
                chunks.append(f"Identification Source: {product['source']} (Confidence: {product['confidence']:.2f})")
        
        # Add image features information
        image_features = structured_content.get("image_features", {})
        if image_features:
            dominant_colors = image_features.get("dominant_colors", {})
            if dominant_colors.get("names"):
                chunks.append(f"Dominant Colors: {', '.join(dominant_colors['names'])}")
            
            shapes = image_features.get("shapes", [])
            if shapes:
                chunks.append(f"Detected Shapes: {', '.join(shapes)}")
            
            aspect_ratio = image_features.get("aspect_ratio", 1)
            chunks.append(f"Image Aspect Ratio: {aspect_ratio:.2f}")
            
            brightness = image_features.get("brightness", 0)
            chunks.append(f"Image Brightness: {brightness:.1f}")
        
        return chunks

# Initialize the image processor
image_processor = ImageProcessor()
