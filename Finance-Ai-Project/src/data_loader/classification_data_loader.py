import os
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from datasets import load_dataset
import re

# Load dataset
dataset = load_dataset("sujet-ai/Sujet-Finance-Vision-10k")
df = dataset['train'].to_pandas()

# Save directory
save_dir = 'data/processed/sujet_images_by_class'

#the dataset is not well classified but we will use a regex search to extract unique classes and map
#them to labels which will serve as our training labels.
# Define the categories and their corresponding regex patterns
categories = {
    "Financial Documents": r"(financial|budget|forecast|expenditure|review|statement)",
    "Marketing Documents": r"(marketing|promotion|media|campaign|brand|advertising)",
    "Corporate/Internal Documents": r"(internal|business|memo|corporate|proposal)",
    "Project Documents": r"(project|estimate|development|testing|costing)",
    "Research Documents": r"(research|experimental|study|test)"
}

# Function to classify document type based on regex match
def classify_document(description, categories):
    for category, regex in categories.items():
        if re.search(regex, description, re.IGNORECASE):
            return category
    return "Uncategorized"  # If no match is found

def decode_and_save_image(row):
    # Classify the document type based on the 'document_type' column using regex
    label = classify_document(row['document_type'], categories)
    
    # Use the label directly as the folder name
    class_dir = os.path.join(save_dir, label)
    
    # Create the directory if it doesn't exist
    os.makedirs(class_dir, exist_ok=True)

    # Decode and save image
    decoded = base64.b64decode(row['encoded_image'])
    image = Image.open(BytesIO(decoded))
    
    # Save image using the unique filename
    image_path = os.path.join(class_dir, f"{row.name}.jpg")
    image.convert("RGB").save(image_path, "JPEG")

# Apply to all rows
df.apply(decode_and_save_image, axis=1)

# Optionally: Keep only relevant columns
df = df[['doc_id', 'content', 'document_type', 'key_details', 'insights']]
print("✅ All images saved and DataFrame cleaned.")
