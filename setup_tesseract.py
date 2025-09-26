"""
Tesseract OCR Setup Script for RAG Narok
This script helps set up Tesseract OCR for offline image processing
"""

import os
import sys
import subprocess
import platform

def check_tesseract_installation():
    """Check if Tesseract is installed and accessible"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Tesseract OCR is already installed!")
            print(f"Version: {result.stdout.strip()}")
            return True
        else:
            print("❌ Tesseract OCR not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        print("❌ Tesseract OCR not found")
        return False

def install_tesseract_instructions():
    """Provide installation instructions for different operating systems"""
    system = platform.system().lower()
    
    print("\n" + "="*60)
    print("🔧 TESSERACT OCR INSTALLATION INSTRUCTIONS")
    print("="*60)
    
    if system == "windows":
        print("\n🪟 WINDOWS INSTALLATION:")
        print("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Run the installer (tesseract-ocr-w64-setup-5.3.3.20231005.exe)")
        print("3. During installation, make sure to check 'Add to PATH'")
        print("4. Restart your terminal/command prompt")
        print("5. Run this script again to verify installation")
        
    elif system == "linux":
        print("\n🐧 LINUX INSTALLATION:")
        print("Ubuntu/Debian:")
        print("  sudo apt update")
        print("  sudo apt install tesseract-ocr")
        print("\nCentOS/RHEL:")
        print("  sudo yum install tesseract")
        print("\nFedora:")
        print("  sudo dnf install tesseract")
        
    elif system == "darwin":  # macOS
        print("\n🍎 MACOS INSTALLATION:")
        print("Using Homebrew:")
        print("  brew install tesseract")
        print("\nOr download from: https://github.com/tesseract-ocr/tesseract/wiki")
        
    else:
        print(f"\n❓ UNKNOWN SYSTEM: {system}")
        print("Please visit: https://github.com/tesseract-ocr/tesseract/wiki")
    
    print("\n" + "="*60)

def setup_python_tesseract():
    """Setup Python Tesseract configuration"""
    print("\n🔧 PYTHON TESSERACT CONFIGURATION:")
    
    # Try to find Tesseract executable
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract"
    ]
    
    tesseract_path = None
    for path in possible_paths:
        if os.path.exists(path):
            tesseract_path = path
            break
    
    if tesseract_path:
        print(f"✅ Found Tesseract at: {tesseract_path}")
        
        # Create configuration file
        config_content = f'''# Tesseract Configuration for RAG Narok
import pytesseract

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"{tesseract_path}"

# Test configuration
try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract configured successfully! Version: {{version}}")
except Exception as e:
    print(f"❌ Tesseract configuration failed: {{e}}")
'''
        
        with open("tesseract_config.py", "w") as f:
            f.write(config_content)
        
        print("✅ Created tesseract_config.py")
        print("📝 You can import this in your main application if needed")
        
    else:
        print("⚠️ Could not auto-detect Tesseract path")
        print("📝 You may need to manually set the path in your code")

def create_fallback_ocr():
    """Create a fallback OCR system for when Tesseract is not available"""
    fallback_code = '''"""
Fallback OCR System for RAG Narok
Used when Tesseract OCR is not available
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class FallbackOCR:
    """Fallback OCR system when Tesseract is not available"""
    
    def __init__(self):
        self.available = False
        self.setup_available_methods()
    
    def setup_available_methods(self):
        """Setup available OCR methods"""
        available_methods = []
        
        # Check for basic image processing
        try:
            import cv2
            available_methods.append("opencv")
        except ImportError:
            pass
        
        try:
            from PIL import Image
            available_methods.append("pillow")
        except ImportError:
            pass
        
        if available_methods:
            self.available = True
            logger.info(f"Fallback OCR methods available: {available_methods}")
        else:
            logger.warning("No fallback OCR methods available")
    
    def extract_text_basic(self, image_path: str) -> Dict:
        """Basic text extraction using available methods"""
        try:
            # For now, return a placeholder with image analysis
            import os
            from PIL import Image
            
            # Get basic image information
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
            
            return {
                "extracted_text": f"Image analysis: {width}x{height} {mode} {format_name} format. OCR not available - please install Tesseract for text extraction.",
                "confidence": 0,
                "method": "fallback_basic",
                "image_info": {
                    "dimensions": f"{width}x{height}",
                    "mode": mode,
                    "format": format_name
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback OCR error: {e}")
            return {
                "extracted_text": f"Error analyzing image: {str(e)}",
                "confidence": 0,
                "method": "fallback_error"
            }

# Initialize fallback OCR
fallback_ocr = FallbackOCR()
'''
    
    with open("fallback_ocr.py", "w") as f:
        f.write(fallback_code)
    
    print("✅ Created fallback_ocr.py for when Tesseract is not available")

def main():
    """Main setup function"""
    print("🚀 RAG Narok - Tesseract OCR Setup")
    print("="*50)
    
    # Check current installation
    if check_tesseract_installation():
        setup_python_tesseract()
        print("\n🎉 Tesseract OCR is ready to use!")
    else:
        print("\n⚠️ Tesseract OCR not found")
        install_tesseract_instructions()
        create_fallback_ocr()
        print("\n📝 After installing Tesseract, run this script again to verify setup")

if __name__ == "__main__":
    main()

