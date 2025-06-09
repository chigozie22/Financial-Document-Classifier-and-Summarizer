from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import os
import streamlit as st

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

@st.cache_resource
def load_model():
    path = "/home/nonso/ai-multimodal-learning-project/Finance-Ai-Project/outputs/models/pegasus_summarizer_model"
    model = PegasusForConditionalGeneration.from_pretrained(path)
    tokenizer = PegasusTokenizer.from_pretrained(path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def financial_document_summarizer(text: str) -> str:
    model, tokenizer, device = load_model()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        summary_id = model.generate(**inputs, max_new_tokens=64)
        return tokenizer.decode(summary_id[0], skip_special_tokens=True)
