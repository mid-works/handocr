from PIL import Image
import pytesseract
import os

# 1. Set the Tesseract executable path (Required for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# 2. Define the image path
image_path = 'swe.png'

# Check if file exists to avoid 'FileNotFoundError'
if not os.path.exists(image_path):
    print(f"Error: The file '{image_path}' was not found in {os.getcwd()}")
else:
    # 3. Open and process
    img = Image.open(image_path)
    
    text = pytesseract.image_to_string(img, lang='eng')
    print("--- Extracted Text ---")
    print(text)
