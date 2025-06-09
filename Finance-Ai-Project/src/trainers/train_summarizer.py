import pandas as pd
import os
from transformers import PegasusTokenizer
from datasets import Dataset, DatasetDict
from transformers import PegasusForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import matplotlib.pyplot as plt
import torch


#edit to suit folder structure
folder_path = "/content/drive/MyDrive/processed/FINDSum/FINDSum"

# train_df = pd.concat(pd.read_csv(os.path.join(folder_path,filename)) for filename in os.listdir(folder_path) if filename.endswith(".csv"))

# train_df.head()[:1].to_dict()


df_1 = pd.read_csv("/content/drive/MyDrive/processed/FINDSum/FINDSum/train_liquidity_segment_0_input_2_1000.csv")
df_2 = pd.read_csv("/content/drive/MyDrive/processed/FINDSum/FINDSum/train_roo_segment_0_input_2_1000.csv")

df = pd.concat([df_1[:600], df_2[:600]])

model_name = "google/pegasus-large"

tokenizer = PegasusTokenizer.from_pretrained(model_name)

def process_inputs(examples):
    model_inputs = tokenizer(
        examples["document"],
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_dataset = Dataset.from_pandas(train_df)
split_dataset = raw_dataset.train_test_split(test_size=0.2)

tokenized_dataset = split_dataset.map(process_inputs, batched=True)

model = PegasusForConditionalGeneration.from_pretrained(model_name)

#training
training_args = Seq2SeqTrainingArguments(
    output_dir="/drive/MyDrive/processed/FINDSum/FINDSum/pegasus_output",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    logging_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=8,
    weight_decay=0.01,
    logging_dir = "/drive/MyDrive/processed/FINDSum/FINDSum/logs",
    load_best_model_at_end=True,
    report_to="none",
    fp16=True,
    )
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator = data_collator,
    compute_metrics=None,
)

trainer.train()
trainer.save_model("/content/drive/MyDrive/processed/FINDSum/FINDSum/models")




#loading my saved model
model = PegasusForConditionalGeneration.from_pretrained("/content/drive/MyDrive/processed/FINDSum/FINDSum/models")
model.eval()

rouge = load_metric("rouge")

import torch
from tqdm import tqdm
def compute_rouge_scores(model, tokenizer, dataset, batch_size=4, max_input_length=512, max_output_length=128):
  model.eval()
  rouge = load_metric("rouge")

  total_preds = []
  total_labels = []

  for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
    inputs = tokenizer(batch["document"], truncation=True, padding=True, max_length=max_input_length, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_masks = inputs["attention_mask"]

    with torch.no_grad():
      outputs = model.generate(
          input_ids=input_ids,
          attention_mask=attention_masks,
          max_length=max_output_length,
          num_beams=4,
      )

    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded_labels = batch["summary"]

    total_preds.extend(decoded_preds)
    total_labels.extend(decoded_labels)

    rouge_scores = rouge.compute(predictions=total_preds, references=total_labels, use_stemmer=True)
    rouge_scores = {k: round(v.mid.fmeasure * 100, 2) for k, v in rouge_scores.items()}
    return rouge_scores
