import streamlit as st
import os 
import tempfile 
from scripts.infer import financial_document_classifier_eval_model
from src.data_loader.classification_data_processor import extract_text_with_docling_or_ocr
from scripts.infer_summarizer import financial_document_summarizer

st.set_page_config(page_title="📄 Financial Document Classifier", layout="wide")

st.title("📄 Financial Document Classifier")
st.markdown("Classify financial documents from **PDFs, images, HTML**, or **raw text input**.")

# === Sidebar UI ===
st.sidebar.header("📂 Upload or Input Text")

uploaded_file = st.sidebar.file_uploader("Upload file", type=["pdf", "jpg", "jpeg", "png", "html", "htm"])

text_input = st.text_area("Or paste raw text here", height=200)
classify_button= st.sidebar.button("🚀 Classify")
summarize_button = st.sidebar.button("Summarize")

# === Main Output Area ===
if classify_button:
    if uploaded_file is not None:
        #save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name 
        extracted_text = extract_text_with_docling_or_ocr(tmp_path)

        os.remove(tmp_path)

        input_source =  f"📤 Uploaded file: {uploaded_file.name}"
    
    elif text_input.strip():
        extracted_text = text_input 
        input_source = "📝 Raw text input"
    else:
        st.warning("Please upload a file or enter some text.")
        st.stop()

    #show text and classification
    st.subheader("📌 Input Source")
    st.info(input_source)

    st.subheader("📜Extracted Text")
    st.text_area("Text Preview", extracted_text[:1500], height=200)

    st.subheader("🔍 Classification Result")
    predicted_class, probabilities = financial_document_classifier_eval_model(extracted_text)

    st.success(f"Predicted Class:**{predicted_class}")

    st.subheader("📊 Class Probabilities")
    st.bar_chart(probabilities)

if summarize_button:
    with st.spinner("Generating Summary:"):
        if uploaded_file is not None:
            #save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name 
            extracted_text = extract_text_with_docling_or_ocr(tmp_path)

            os.remove(tmp_path)

            input_source =  f"📤 Uploaded file: {uploaded_file.name}"
        
        elif text_input.strip():
            extracted_text = text_input 
            input_source = "📝 Raw text input"
        else:
            st.warning("Please upload a file or enter some text.")
            st.stop()
        
        #show text and classification
        st.subheader("📌 Input Source")
        st.info(input_source)
    
        try:
            generated_summary = financial_document_summarizer(extracted_text)
        except Exception as e:
            st.error(f"❌ Summary generation failed: {e}")
            st.stop()

        st.success(f"Summary: {generated_summary}")
        st.write(generated_summary)
        