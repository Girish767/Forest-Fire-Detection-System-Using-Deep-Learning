import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    AutoImageProcessor, 
    SiglipForImageClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
from datasets import load_dataset

#configuration and hyperparameters
MODEL_ID = "google/siglip-base-patch16-224" # Base model to fine-tune
OUTPUT_DIR = "./forest_fire_model_v1"
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
DATASET_DIR = "./dataset" 

if not os.path.exists(DATASET_DIR):
    print(f"ERROR: Dataset directory '{DATASET_DIR}' not found.")
    print("Please ensure your dataset is organized as follows:")
    print("dataset/")
    print("  train/")
    print("    Fire/ (images)")
    print("    Smoke/ (images)")
    print("    Normal/ (images)")
    exit(1)

#data augmentation and preprocessing
print("Initializing Data Pre-processing and Augmentation pipelines...")

# Initialize Processor (handles resizing and normalization specific to SigLIP)
processor = AutoImageProcessor.from_pretrained(MODEL_ID)


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), # Zoom in/out
    transforms.RandomHorizontalFlip(p=0.5),                   # Mirror image
    transforms.RandomRotation(degrees=15),                    # Rotate slightly
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Change lighting
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Define Validation Transforms (No augmentation, just resizing/norm)
val_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

def transform_train(examples):
    examples["pixel_values"] = [train_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples

def transform_val(examples):
    examples["pixel_values"] = [val_transforms(image.convert("RGB")) for image in examples["image"]]
    return examples

#dataset loading
print("Loading Dataset...")

dataset = load_dataset("imagefolder", data_dir=DATASET_DIR)

#splitting dataset
if "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2)

# Applying transforms
dataset["train"].set_transform(transform_train)
dataset["test"].set_transform(transform_val)

labels = dataset["train"].features["label"].names
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

print(f"Classes found: {labels}")

#model initialization
print(f"Initializing Model: {MODEL_ID}")

model = SiglipForImageClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True # Necessary when fine-tuning on new classes
)

#training metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

#training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    report_to="none" 
)

# initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

#training execution
if __name__ == "__main__":
    print("Starting Training...")
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    print(f"Training completed. Model saved to {OUTPUT_DIR}")
