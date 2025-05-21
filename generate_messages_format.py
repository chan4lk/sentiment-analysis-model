import json
import random
from faker import Faker
import math
import os
from typing import List, Dict

fake = Faker()

# --- Configuration ---
NUM_RECORDS_TOTAL = 1000
OUTPUT_DIR = "data/jsonl"
TRAIN_FILENAME = "train_messages.jsonl"
VALID_FILENAME = "valid_messages.jsonl"
TEST_FILENAME = "test_messages.jsonl"

# Define split ratios (adjust if needed)
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
# TEST_RATIO implicitly becomes 1.0 - TRAIN_RATIO - VALID_RATIO

SYSTEM_PROMPT = "you are a sentiment analysis agent"
INSTRUCTION_PROMPT = "Classify the sentiment (Positive, Negative, or Neutral) for this ServiceNow ticket text:"

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
            text = text.replace(placeholder, generator(), 1)
    return text

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
        "Inquiry about status of [PROJECT_NAME] project.",
        "Submitting request for software license for [APP].",
        "Following up on ticket [TICKET_ID].",
        "User [USER] reporting issue with [DEVICE] at [LOCATION].",
        "Please provide info on MFA setup.",
        "VM provisioning request.",
        "Work Note: Checked logs on [SERVER]. Assigning to [GROUP].",
        "Additional Comment: Any update on this request?",
        "Resolution Note: Restarted service on [SERVER]. User confirmed fix.",
    ]
}

def make_message_record(text_content: str, sentiment: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INSTRUCTION_PROMPT}\n{text_content}"},
        {"role": "assistant", "content": sentiment}
    ]

all_records = []
records_generated = 0

while records_generated < NUM_RECORDS_TOTAL:
    rand_choice = random.random()
    if rand_choice < 0.40:
        sentiment = "Negative"
    elif rand_choice < 0.85:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    text_pattern = random.choice(SENTIMENT_PATTERNS[sentiment])
    text_content = fill_placeholders(text_pattern)\
        .replace("[PROJECT_NAME]", fake.bs().title())\
        .replace("[TICKET_ID]", f"{random.choice(['INC','REQ'])}{random.randint(10000,99999)}")
    messages = make_message_record(text_content, sentiment)
    all_records.append({"messages": messages})
    records_generated += 1

random.shuffle(all_records)
num_train = int(NUM_RECORDS_TOTAL * TRAIN_RATIO)
num_valid = int(NUM_RECORDS_TOTAL * VALID_RATIO)
train_records = all_records[:num_train]
valid_records = all_records[num_train : num_train + num_valid]
test_records = all_records[num_train + num_valid :]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def write_jsonl(filepath: str, records: List[Dict]):
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

write_jsonl(os.path.join(OUTPUT_DIR, TRAIN_FILENAME), train_records)
write_jsonl(os.path.join(OUTPUT_DIR, VALID_FILENAME), valid_records)
write_jsonl(os.path.join(OUTPUT_DIR, TEST_FILENAME), test_records)

print("Message-format dataset generation complete.")
print(f"Files created: {TRAIN_FILENAME}, {VALID_FILENAME}, {TEST_FILENAME} in '{OUTPUT_DIR}' directory.")
