import json
import random
from faker import Faker
import math
import os

fake = Faker()

# --- Configuration ---
NUM_RECORDS_TOTAL = 1000
OUTPUT_DIR = "data"
TRAIN_FILENAME = "train.jsonl"
VALID_FILENAME = "valid.jsonl" # Using .jsonl for consistency
TEST_FILENAME = "test.jsonl"

# Define split ratios (adjust if needed)
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
# TEST_RATIO implicitly becomes 1.0 - TRAIN_RATIO - VALID_RATIO

INSTRUCTION_PROMPT = "Classify the sentiment (Positive, Negative, or Neutral) for this ServiceNow ticket text:"

# --- Content Templates & Keywords (Simplified from previous example) ---
PLACEHOLDERS = {
    "user": fake.name,
    "app": lambda: random.choice(["SAP", "Outlook", "Teams", "VPN", "Salesforce", "Portal"]),
    "device": lambda: random.choice(["laptop", "monitor", "printer", "phone"]),
    "server": lambda: f"SRV-{fake.word().upper()}-{random.randint(1, 9):01d}",
    "location": fake.city,
    "error_code": lambda: f"E{random.randint(100, 999)}",
    "group": lambda: random.choice(["Network", "HelpDesk", "ServerTeam", "AppSupport"]),
}

def fill_placeholders(text):
    for key, generator in PLACEHOLDERS.items():
        placeholder = f"[{key.upper()}]"
        while placeholder in text:
            # Use loop to replace multiple occurrences if any
            text = text.replace(placeholder, generator(), 1)
    return text

# Simplified patterns mapped directly to sentiment
SENTIMENT_PATTERNS = {
    "Negative": [
        "Cannot login to [APP]. Error [ERROR_CODE]. Urgent!",
        "My [DEVICE] is broken. Need replacement.",
        "[APP] extremely slow today.",
        "System crash on [SERVER]. Lost work!",
        "Still waiting for update from yesterday. Very frustrating.",
        "Solution did not work. Problem persists.",
        "Network connection keeps dropping in [LOCATION].",
        "Unacceptable performance for [APP].",
    ],
    "Positive": [
        "Thanks! [APP] working perfectly now.",
        "Issue resolved quickly by [GROUP]. Appreciate the help!",
        "New [DEVICE] works great.",
        "Problem solved after following your steps.",
        "Confirmed fix is working. Closing ticket now.",
        "Great job resolving the outage!",
        "Access granted. Thank you!",
        "Excellent support, problem fixed.",
    ],
    "Neutral": [
        "Requesting access to [APP] for new user [USER].",
        "Need standard [DEVICE] for new employee.",
        "Password reset required for [USER].",
        "Inquiry about status of [PROJECT_NAME] project.", # Needs placeholder
        "Submitting request for software license for [APP].",
        "Following up on ticket [TICKET_ID].", # Needs placeholder
        "User [USER] reporting issue with [DEVICE] at [LOCATION].",
        "Please provide info on MFA setup.",
        "VM provisioning request.",
        "Work Note: Checked logs on [SERVER]. Assigning to [GROUP].",
        "Additional Comment: Any update on this request?",
        "Resolution Note: Restarted service on [SERVER]. User confirmed fix.",
    ]
}

# --- Generation Logic ---
all_records = []
records_generated = 0

print(f"Generating {NUM_RECORDS_TOTAL} synthetic records...")

while records_generated < NUM_RECORDS_TOTAL:
    # Choose a sentiment category somewhat randomly
    rand_choice = random.random()
    if rand_choice < 0.40:
        sentiment = "Negative"
    elif rand_choice < 0.85:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    # Select a random pattern for that sentiment
    text_pattern = random.choice(SENTIMENT_PATTERNS[sentiment])

    # Fill placeholders (add simple ones for patterns needing them)
    text_content = fill_placeholders(text_pattern)\
                   .replace("[PROJECT_NAME]", fake.bs().title())\
                   .replace("[TICKET_ID]", f"{random.choice(['INC','REQ'])}{random.randint(10000,99999)}")

    # Construct the text payload in the desired format
    mlx_text_payload = f"<|user|>\n{INSTRUCTION_PROMPT}\n{text_content} <|end|>\n<|assistant|> \n{sentiment} <|end|>"

    # Create the JSON object
    record = {"text": mlx_text_payload}
    all_records.append(record)
    records_generated += 1

print(f"Generated {len(all_records)} records.")

# --- Shuffle and Split Data ---
print("Shuffling and splitting data...")
random.shuffle(all_records)

# Calculate split points
num_train = int(NUM_RECORDS_TOTAL * TRAIN_RATIO)
num_valid = int(NUM_RECORDS_TOTAL * VALID_RATIO)
# num_test is the remainder

train_records = all_records[:num_train]
valid_records = all_records[num_train : num_train + num_valid]
test_records = all_records[num_train + num_valid :]

print(f"Train set size: {len(train_records)}")
print(f"Validation set size: {len(valid_records)}")
print(f"Test set size: {len(test_records)}")

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory '{OUTPUT_DIR}' created or already exists.")

# --- Write Files ---
def write_jsonl(filepath, records):
    """Helper function to write a list of records to a JSON Lines file."""
    with open(filepath, 'w') as f:
        for record in records:
            json.dump(record, f)
            f.write('\n')

train_filepath = os.path.join(OUTPUT_DIR, TRAIN_FILENAME)
valid_filepath = os.path.join(OUTPUT_DIR, VALID_FILENAME)
test_filepath = os.path.join(OUTPUT_DIR, TEST_FILENAME)

print(f"Writing {TRAIN_FILENAME}...")
write_jsonl(train_filepath, train_records)

print(f"Writing {VALID_FILENAME}...")
write_jsonl(valid_filepath, valid_records)

print(f"Writing {TEST_FILENAME}...")
write_jsonl(test_filepath, test_records)

print("\nDataset generation and splitting complete.")
print(f"Files created in directory: '{os.path.abspath(OUTPUT_DIR)}'")

# --- Optional: Print a sample from train file ---
print("\n--- Sample line from train.jsonl ---")
try:
    with open(train_filepath, 'r') as f_read:
        print(f_read.readline().strip())
except FileNotFoundError:
    print(f"Error: Could not read back file {train_filepath}")