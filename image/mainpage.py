import cv2
import pytesseract
import os

# Set Tesseract executable path
#pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Lenovo\Downloads\Tesseract-OCR\tesseract.exe'

# Set TESSDATA_PREFIX
#os.environ['TESSDATA_PREFIX'] = 'C:\Users\Lenovo\OneDrive\Documents\tess\tessdata'

# Load the image
# Use a raw string to avoid the 'unicodeescape' error
image_path = r"C:\Users\Lenovo\OneDrive\Desktop\Project1\image\tamil.webp"

image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load the image.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Extract text with options
    extracted_text = pytesseract.image_to_string(binary_image, lang='tam', config='--psm 6')

    # Display the extracted text
    print("Extracted Text:")
    print(extracted_text)
