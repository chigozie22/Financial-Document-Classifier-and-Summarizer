from bs4 import BeautifulSoup
import os 
import pytesseract
from PIL import Image
import pandas as pd
from paddleocr import PaddleOCR
from docling.document_converter import DocumentConverter


# Initialize PaddleOCR once (supports French, English, etc.)
  # Use 'fr' for French if needed

def extract_text_from_html(file_path):
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"❌ BeautifulSoup failed for {file_path}: {e}")
        return ""

def extract_text_with_docling_or_ocr(file_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    # Handle HTML separately with BeautifulSoup
    if file_path.lower().endswith(('.html', '.htm')):
        return extract_text_from_html(file_path)

    # Try Docling
    try:
        converter = DocumentConverter()
        doc = converter.convert(file_path)
        if hasattr(doc, 'text') and doc.text.strip():
            return doc.text
        
    except Exception as e:
        print(f"❌ Docling failed for {file_path}: {e}")

    # Fallback to PaddleOCR for images
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"🔁 Falling back to PaddleOCR for {file_path}")
        try:
            result = ocr.ocr(file_path, cls=True)
            extracted_text = ""
            for line in result:
                for box in line:
                    extracted_text += box[1][0] + "\n"
            return extracted_text.strip()
        except Exception as e:
            print(f"❌ PaddleOCR also failed for {file_path}: {e}")

    #if docling fails to extract pdf
    if file_path.lower().endswith(".jpg"):
        print(f"Falling back to paddleocr for pdf {file_path}")
        try:
            import fitz
            pdf_doc = fitz.open(file_path)
            all_text = ""
            for page in pdf_doc:
                all_text += page.get_text()
            pdf_doc.close()
            return all_text.strip()
        except Exception as e:
            print(f"PaddleOCR failed for pdf {file_path}: {e}")
            
    return ""




def process_documents_to_text(root_dir):
    rows = []

    for folder_name in os.listdir(root_dir):
        label = folder_name
        folder_path = os.path.join(root_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if file_path.lower().endswith(('.html', '.htm', '.jpg', '.jpeg', '.png')):
                text = extract_text_with_docling_or_ocr(file_path)
                if text:
                    rows.append({
                        "text": text,
                        "label": label
                    })

    return pd.DataFrame(rows)

def process_documents_for_inference(text):
    if os.path.isfile(text) and text.lower().endswith(('.html', '.htm', '.jpg', '.jpeg', '.png', '.pdf')):
        output = extract_text_with_docling_or_ocr(text)
        return output
    
    return text.strip()

def save_file_to_csv(file_path):

    df = process_documents_to_text(file_path)
        # "/home/nonso/ai-multimodal-learning-project/Finance-Ai-Project/data/processed/sujet_images_by_class")

    df.to_csv("finance_text_dataset.csv", index=False)
    print("✅ Dataset saved with", len(df), "entries.")
    print(df.head())
