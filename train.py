import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, # <-- Comment out or remove this if not used elsewhere
    Phi3ForSequenceClassification,       # <-- Add this import
    TrainingArguments,
    Trainer,
    # BitsAndBytesConfig is removed as it's not supported on MPS
)
import numpy as np
import os
import math # Make sure math is imported at the top


# --- 1. Configuration ---
# MODEL_CHECKPOINT = "microsoft/phi-3-mini-4k-instruct" # <-- Comment out/Remove Phi-3

# Option 1 (Recommended Start):
MODEL_CHECKPOINT = "distilbert-base-uncased"

# Option 2:
# MODEL_CHECKPOINT = "bert-base-uncased"

# Option 3:
# MODEL_CHECKPOINT = "roberta-base"

# Option 4:
# MODEL_CHECKPOINT = "albert-base-v2"
# MODEL_CHECKPOINT = "microsoft/phi-3-mini-128k-instruct" # Alternative if you need longer context
CSV_FILE_PATH = "synthetic_servicenow_sentiment_1000.csv"  # Your dataset file
TEXT_COLUMN = "text_content"          # Column with text to classify
LABEL_COLUMN = "sentiment"            # Column with sentiment labels
OUTPUT_DIR = "./sentiment-finetuned-mps" # Where to save the model
NUM_LABELS = 3                       # Positive, Negative, Neutral
MAX_TOKEN_LENGTH = 512               # Max sequence length for tokenizer (Might need to lower for memory on Mac)
TEST_SIZE = 0.15                     # Proportion for validation and test sets
VALIDATION_SIZE = 0.15               # Proportion of original data for validation
RANDOM_SEED = 42

# --- Training Hyperparameters (Adjust based on your Mac's performance and memory) ---
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 1                # Start SMALLER on Mac M3 unified memory
EVAL_BATCH_SIZE = 2                 # Start SMALLER on Mac M3 unified memory
NUM_EPOCHS = 2                      # Start with 1-3 epochs for fine-tuning
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4     # INCREASE to compensate for small batch sizes
USE_QLORA = False                   # <<<< IMPORTANT: QLoRA (bitsandbytes) is NOT supported on MPS
USE_FP16 = False                     # <<<< Enable FP16 for potential speedup/memory saving on MPS

# --- Check for MPS (Apple Silicon GPU) ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
    # FP16 is generally supported, BF16 less so or not performant on MPS
    compute_dtype = torch.float16 if USE_FP16 else torch.float32
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU (Training will be very slow)")
    USE_FP16 = False # FP16 not applicable on CPU
    compute_dtype = torch.float32

# --- 2. Load and Prepare Data ---
print(f"Loading data from {CSV_FILE_PATH}...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    # Basic cleaning - remove rows with missing text or labels
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Map labels to integers ---
unique_labels = df[LABEL_COLUMN].unique()
if len(unique_labels) != NUM_LABELS:
     print(f"Warning: Expected {NUM_LABELS} unique labels, but found {len(unique_labels)}: {unique_labels}")
     # Handle unexpected label count if necessary

label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df[LABEL_COLUMN].map(label2id)

print(f"Label mapping: {label2id}")

# --- Split Data ---
print("Splitting data into train, validation, and test sets...")
train_df, temp_df = train_test_split(
    df,
    test_size=TEST_SIZE + VALIDATION_SIZE,
    random_state=RANDOM_SEED,
    stratify=df['label']
)
validation_df, test_df = train_test_split(
    temp_df,
    test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE),
    random_state=RANDOM_SEED,
    stratify=temp_df['label']
)

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(validation_df)}")
print(f"Test set size: {len(test_df)}")

# --- Convert to Hugging Face Dataset ---
train_dataset = Dataset.from_pandas(train_df[[TEXT_COLUMN, 'label']])
validation_dataset = Dataset.from_pandas(validation_df[[TEXT_COLUMN, 'label']])
test_dataset = Dataset.from_pandas(test_df[[TEXT_COLUMN, 'label']])

raw_datasets = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

# --- 3. Load Tokenizer ---
print(f"Loading tokenizer for {MODEL_CHECKPOINT}...")
# trust_remote_code is needed for Phi-3
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, trust_remote_code=True)

# --- Set padding token if missing ---
if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
    tokenizer.pad_token = tokenizer.eos_token # Common practice
    # Important: Update model config later if needed *after* loading

# --- Tokenize Data ---
def tokenize_function(examples):
    return tokenizer(
        examples[TEXT_COLUMN],
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKEN_LENGTH
    )

print("Tokenizing datasets...")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns([TEXT_COLUMN])
tokenized_datasets.set_format("torch")

# --- 4. Configure Model (NO QLoRA on MPS) ---
print(f"Loading model {MODEL_CHECKPOINT} for sequence classification...")
print("NOTE: QLoRA (4-bit quantization) is disabled as it's not supported on MPS.")

# --- Calculate steps per epoch ---
# Ensure tokenized_datasets is defined before this point
num_train_samples = len(tokenized_datasets["train"])
effective_batch_size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
steps_per_epoch = math.ceil(num_train_samples / effective_batch_size)
print(f"Calculated steps per epoch: {steps_per_epoch}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    # trust_remote_code=True, # Still required for Phi-3 architecture
    # No quantization_config or device_map here
    # torch_dtype=compute_dtype # Optionally specify dtype at load time, though Trainer handles it too
    ignore_mismatched_sizes=True # Add this if you encounter size mismatch errors after changing class
)

# Ensure the pad token ID is consistent between tokenizer and model config
if tokenizer.pad_token_id is not None:
     model.config.pad_token_id = tokenizer.pad_token_id
else:
     print("Warning: Pad token ID is None after model loading.")
     if tokenizer.eos_token_id is not None:
          print(f"Setting model's pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
          model.config.pad_token_id = tokenizer.eos_token_id

# No explicit model.to(device) needed here; Trainer handles it.

# --- 5. Define Evaluation Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 6. Define Training Arguments ---
print("Configuring training arguments for MPS...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    # evaluation_strategy="epoch",  # Remove this line
    eval_strategy="epoch",
    save_strategy="epoch",          # Keep this for saving every epoch
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,               # Keep or adjust logging frequency
    eval_steps=steps_per_epoch,     # <--- Add this line
    fp16=USE_FP16 if device.type == 'mps' else False,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    seed=RANDOM_SEED,
    optim="adamw_torch",
    use_mps_device= (device.type == 'mps'),
)


# --- 7. Initialize Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --- 8. Train the Model ---
print("Starting model training on MPS/CPU...")
print("Expect this to be potentially slower and use more memory than CUDA with QLoRA.")
try:
    train_result = trainer.train()
    print("Training finished.")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # --- 9. Evaluate on Test Set ---
    print("\nEvaluating model on the test set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Test Set Evaluation Results:")
    print(test_metrics)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    # --- 10. Save the Final Model & Tokenizer ---
    print(f"\nSaving the fine-tuned model and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("\nFine-tuning process complete.")
    print(f"Model saved to: {os.path.abspath(OUTPUT_DIR)}")

except Exception as e:
     # Catch generic exceptions; specific memory errors might differ on MPS vs CUDA
     print(f"\nAn error occurred during training or evaluation: {e}")
     print("If you encountered memory issues, try further reducing `TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`,")
     print("or `MAX_TOKEN_LENGTH`. Increase `GRADIENT_ACCUMULATION_STEPS` to compensate for smaller batch sizes.")