# 💼 Financial Document Classification and Summarization using Transformers

This project applies modern NLP techniques to financial documents through two key tasks:

1. **Text Classification** – Classify financial reports into categories like liquidity, risk, operations, etc., using FinBERT.
2. **Summarization** – Generate concise and informative summaries of long financial documents using PEGASUS fine-tuned on the FINDSum dataset.

---

## 🚀 Project Overview

- **Task 1**: Financial Document Classification  
  - Model: [`yiyanghkust/finbert-pretrain`](https://huggingface.co/yiyanghkust/finbert-pretrain)  
  - Dataset: sujet-ai/Sujet-Finance-Vision-10k  financial documents downloaded from huggingface and labeled using regex mapping

- **Task 2**: Financial Document Summarization  
  - Model: [`google/pegasus-large`](https://huggingface.co/google/pegasus-large)  
  - Dataset: [FINDSum](https://huggingface.co/datasets/findsum)

---

## 🗂️ Datasets

### 🧾 1. Classification Dataset

Used to train FinBERT on multi-class classification of financial documents. Classes may include:
- "Balance Sheets",
- "Cash Flow",
- "Corporate_Internal Documents",
- "Financial Documents",
- "Income Statement",
- "Marketing Documents",
- "Project Documents",
- "Research Documents"
        

CSV format:
