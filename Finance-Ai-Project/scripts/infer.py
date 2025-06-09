from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from src.data_loader.classification_data_processor import process_documents_for_inference
import os



#evaluating the model
def financial_document_classifier_eval_model(text: str) -> str:
    
    

    path = "/home/nonso/ai-multimodal-learning-project/Finance-Ai-Project/outputs/models/finbert-financial-classifier"

    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")

    model.eval()

    #tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    classifier_labels = [
        "Balance Sheets",
        "Cash Flow",
        "Corporate_Internal Documents",
        "Financial Documents",
        "Income Statement",
        "Marketing Documents",
        "Project Documents",
        "Research Documents",
    ]
        
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        predicted_index= torch.argmax(probs, dim=-1)
        predicted_label = classifier_labels[predicted_index]

        return predicted_label, {label: prob.item() for label, prob in zip(classifier_labels, probs[0])}
        