import os
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F


#load the dataset extracted using the text_extractor function
#adjust file director 
df = pd.read_csv("/content/drive/MyDrive/sujet_images_by_class/finance_data.csv")

#to encode the labels
#encode the labels
le = LabelEncoder()
le.fit_transform(df["label"])
# df["label"] = le.fit_transform(df["label"])

#convert the dataset to huggingface dataset
#convert to hugging face dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

#tokenization
model_checkpoint = "yiyanghkust/finbert-pretrain"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize(batch):
    return tokenizer(batch["text"], max_length=512, padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)


num_labels = df["label"].nunique()

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


#to compute metrics 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_score": f1_score(labels, preds, average="weighted")
    }

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/models/finbert_classifier",
    eval_strategy= "epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="/content/drive/MyDrive/models/logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_score",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

#saving the model
trainer.save_model("/content/drive/MyDrive/models/financial_finbert_classfier")

path = "/content/drive/MyDrive/models/financial_finbert_classfier"

model = AutoModelForSequenceClassification.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")

#evaluating the model
def eval_model(text):
    model.eval()
    #tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    #inference
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
        predicted_index= torch.argmax(probs, dim=-1).item()
        class_name = classifier_labels[predicted_index]

 
    print(f"Class probabilites: {probs}")
    print(f"Predicted class name: {class_name}")
