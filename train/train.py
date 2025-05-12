import json
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import torch

import mlflow
import mlflow.transformers

mlflow.set_experiment("bert-mlm-papers")

# Prepare data
json_path = "/mnt/object/trainingdata/train.json"
text_path = "train.txt"

if not os.path.exists(text_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(text_path, "w", encoding="utf-8") as f:
        for item in data:
            text = item["text"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")

dataset = load_dataset("text", data_files={"train": text_path})

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="/mnt/block/bart_model",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none"
)

# Custom callback to log metrics to MLflow at each logging step
class MLflowLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            mlflow.log_metrics(logs, step=state.global_step)

# Start MLflow run
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params({
        "model_name": model_name,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "mlm_probability": 0.15,
        "max_length": 512,
        "learning_rate": training_args.learning_rate,
        "fp16": training_args.fp16,
    })

    # Log system info (optional)
    try:
        if torch.cuda.is_available():
            gpu_info = os.popen("nvidia-smi").read()
            mlflow.log_text(gpu_info, "gpu-info.txt")
    except Exception as e:
        print("Could not log GPU info:", e)

    # Trainer with MLflow callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
        callbacks=[MLflowLoggingCallback()],
    )

    trainer.train()

    # Save model and tokenizer to block storage
    trainer.save_model("/mnt/block/bart_model")
    tokenizer.save_pretrained("/mnt/block/bart_model")

    # Log the model as an MLflow artifact
    mlflow.transformers.log_model(
        transformers_model=trainer.model,
        artifact_path="bert-mlm-model",
        tokenizer=tokenizer,
        input_example=tokenizer("Example input for MLflow logging", return_tensors="pt")
    )

    print("Fine-tuning complete! Model saved to /mnt/block/bart_model and logged to MLflow.")
