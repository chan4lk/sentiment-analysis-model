import pandas as pd
import random
from faker import Faker
import math

fake = Faker()

# --- Configuration ---
NUM_RECORDS_TARGET = 1000
OUTPUT_CSV_FILE = "synthetic_servicenow_sentiment_1000.csv"

# --- Content Templates & Keywords ---
PLACEHOLDERS = {
    "user": fake.name,
    "app": lambda: random.choice(["SAP", "Outlook", "Teams", "VPN Client", "Salesforce", "Internal Portal", "Finance Tool", "HR System"]),
    "device": lambda: random.choice(["laptop", "monitor", "keyboard", "mouse", "docking station", "printer", "desk phone"]),
    "server": lambda: f"{random.choice(['APP', 'DB', 'WEB', 'DC'])}-{fake.word().upper()}-{random.randint(1, 99):02d}",
    "location": fake.city,
    "error_code": lambda: f"{random.choice(['E', 'W', 'C'])}{random.randint(100, 9999)}",
    "group": lambda: random.choice(["Network Team", "Help Desk", "Server Support", "App Support Team", "Security Ops", "Desktop Support"]),
}

def fill_placeholders(text):
    for key, generator in PLACEHOLDERS.items():
        placeholder = f"[{key.upper()}]"
        # Use a loop in case a placeholder appears multiple times
        while placeholder in text:
            text = text.replace(placeholder, generator(), 1)
    return text

# Sentiment-associated phrases (simplified)
NEGATIVE_PATTERNS = [
    "Cannot login to [APP]. Getting error [ERROR_CODE]. Urgent!",
    "My [DEVICE] is broken. Need a replacement ASAP.",
    "[APP] is extremely slow today. Taking minutes to load.",
    "System crash on [SERVER]. Lost work. Please investigate immediately.",
    "Still waiting for an update on my issue from yesterday.",
    "The provided solution did not work. Problem persists.",
    "Error [ERROR_CODE] when trying to access the shared drive.",
    "Network connection keeps dropping in [LOCATION] office.",
    "This is unacceptable performance for [APP].",
    "Frustrated with the constant issues with the [DEVICE].",
]
POSITIVE_PATTERNS = [
    "Thanks! [APP] is working perfectly now.",
    "Issue resolved quickly. Appreciate the help from [GROUP].",
    "The new [DEVICE] works great. Setup was smooth.",
    "Problem solved after the steps you provided.",
    "Confirmed fix is working. Closing the ticket.",
    "Great job on resolving the outage so fast!",
    "Access granted. Thank you for the prompt response.",
    "Finally working again after restart. Thanks!",
    "Excellent support, problem fixed.",
    "User [USER] confirmed resolution.",
]
NEUTRAL_PATTERNS_DESC = [
    "Requesting access to [APP] for new user [USER].",
    "Need standard [DEVICE] for employee starting next week.",
    "Password reset required for [USER]'s account.",
    "Inquiry about the status of upgrade project '[PROJECT_NAME]'.", # Needs a project name placeholder/generator
    "Submitting request for software license for [APP].",
    "Following up on previous ticket [TICKET_ID].", # Needs a ticket id placeholder
    "User [USER] reporting issue with [DEVICE] at [LOCATION].",
    "Requesting information on how to configure MFA.",
    "Hardware refresh needed for [USER]. See attached list.",
    "VM provisioning request based on template '[TEMPLATE_NAME]'.", # Needs template name
]
NEUTRAL_PATTERNS_COMMENT_WN = [ # Work Notes
    "Checked logs on [SERVER]. Found error [ERROR_CODE].",
    "Assigned ticket to [GROUP].",
    "Attempted remote connection to user's machine. Failed.",
    "Gathering more information from user [USER].",
    "Escalating issue to Tier 3 support / [GROUP].",
    "Monitoring system performance after patch deployment.",
    "Ordered replacement [DEVICE]. ETA 2 days.",
    "Reset password for [USER] and provided temporary one.",
    "Investigating network latency between [LOCATION] and DC.",
    "Applied configuration change as per documentation.",
]
NEUTRAL_PATTERNS_COMMENT_AC = [ # Additional Comments (often neutral status checks)
    "Any update on this request?",
    "Just checking in on the status.",
    "Following up as requested.",
    "Can you please provide an estimated resolution time?",
    "Is there any further information needed from my side?",
]
NEUTRAL_PATTERNS_RES = [ # Resolution Notes (often neutral factual)
    "Resolved by restarting the [APP] service on [SERVER].",
    "User [USER] confirmed issue resolved after clearing cache.",
    "Provided user with documentation link. Closing ticket.",
    "Access granted to [APP]. User notified.",
    "Replaced faulty [DEVICE]. Asset tag updated.",
    "Password reset completed. User able to login.",
    "Root cause identified as [CAUSE]. Permanent fix tracked in PRB[NUMBER].", # Needs cause/number
    "Software installed and verified with user.",
    "Network configuration corrected by [GROUP].",
    "Closed per user request.",
]

# --- Generation Logic ---
data = []
ticket_counter_inc = random.randint(10000, 20000)
ticket_counter_req = random.randint(50000, 60000)
entry_id_counter = 1

while len(data) < NUM_RECORDS_TARGET:
    is_incident = random.random() < 0.65 # More incidents than requests

    if is_incident:
        ticket_id = f"INC{ticket_counter_inc}"
        ticket_counter_inc += 1
        # Simulate shorter resolution times for incidents (highly variable)
        resolution_time = random.randint(15, 1440) # 15 mins to 24 hours
        num_entries = random.randint(3, 6) # Incidents often have more back-and-forth
    else:
        ticket_id = f"REQ{ticket_counter_req}"
        ticket_counter_req += 1
        # Simulate potentially longer resolution times for requests
        resolution_time = random.randint(60, 4320) # 1 hour to 3 days
        num_entries = random.randint(2, 4) # Requests can be simpler

    current_ticket_entries = []

    # 1. Short Description
    sdesc_sentiment = "Neutral"
    sdesc_text = ""
    if is_incident:
        if random.random() < 0.7: # Incident short desc often implies negativity
             sdesc_sentiment = "Negative"
             sdesc_text = fill_placeholders(random.choice([
                 f"[APP] Down", f"Cannot Login to [APP]", f"[DEVICE] Broken",
                 f"Network Slow in [LOCATION]", f"Urgent: [APP] Access Issue"
             ]))
        else:
             sdesc_sentiment = "Neutral" # Sometimes just factual
             sdesc_text = fill_placeholders(random.choice([
                 f"Issue with [DEVICE]", f"Login Problem [APP]", f"Connectivity Error"
             ]))
    else: # Request
        sdesc_sentiment = "Neutral"
        sdesc_text = fill_placeholders(random.choice([
            f"Request for [APP] Access", f"New [DEVICE] Request", f"Password Reset",
            f"Software Installation [APP]", f"Information Request"
        ]))

    current_ticket_entries.append({
        "entry_id": entry_id_counter, "ticket_ref_id": ticket_id, "source_field": "short_description",
        "text_content": sdesc_text, "sentiment": sdesc_sentiment, "resolution_time_minutes": resolution_time
    })
    entry_id_counter += 1

    # 2. Description
    desc_sentiment = "Neutral"
    desc_text = ""
    if is_incident:
        desc_sentiment = "Negative" # Usually negative detail
        desc_text = fill_placeholders(random.choice(NEGATIVE_PATTERNS))
    else: # Request
        desc_sentiment = "Neutral"
        # Add placeholders needed by NEUTRAL_PATTERNS_DESC here if desired
        desc_text = fill_placeholders(random.choice(NEUTRAL_PATTERNS_DESC)
                                      .replace("[PROJECT_NAME]", fake.bs().title())
                                      .replace("[TICKET_ID]", f"INC{random.randint(10000, ticket_counter_inc-1)}")
                                      .replace("[TEMPLATE_NAME]", f"{random.choice(['Dev','Test','Standard'])} VM Template")
                                      )

    current_ticket_entries.append({
        "entry_id": entry_id_counter, "ticket_ref_id": ticket_id, "source_field": "description",
        "text_content": desc_text, "sentiment": desc_sentiment, "resolution_time_minutes": resolution_time
    })
    entry_id_counter += 1


    # 3. Comments (Work Notes & Additional Comments)
    num_comments = num_entries - 3 # Adjusted based on total entries desired
    if num_comments < 0: num_comments = 0
    comment_sentiments = [] # Track sentiments to potentially influence resolution/final comment

    for i in range(num_comments):
        is_worknote = random.random() < 0.6 # More likely a work note
        comment_sentiment = "Neutral"
        comment_text = ""

        if is_worknote:
            comment_text = fill_placeholders(random.choice(NEUTRAL_PATTERNS_COMMENT_WN))
        else: # Additional Comment from user
            # Make user comments sometimes negative, especially if incident isn't resolved
            if is_incident and i > 0 and random.random() < 0.4: # More likely negative if it's later in an incident
                comment_sentiment = "Negative"
                comment_text = fill_placeholders(random.choice([
                    "Any update? Still blocked.", "This is taking too long.", "Is anyone working on this??",
                    "This impacts deadline [DATE]. Need resolution!", "No progress reported yet."
                ]).replace("[DATE]", fake.future_date(end_date="+7d").strftime('%Y-%m-%d')))
            else:
                comment_sentiment = "Neutral" # Or just neutral status check
                comment_text = fill_placeholders(random.choice(NEUTRAL_PATTERNS_COMMENT_AC))

        comment_sentiments.append(comment_sentiment)
        current_ticket_entries.append({
            "entry_id": entry_id_counter, "ticket_ref_id": ticket_id, "source_field": "comments",
            "text_content": comment_text, "sentiment": comment_sentiment, "resolution_time_minutes": resolution_time
        })
        entry_id_counter += 1


    # 4. Resolution Notes / Final Comment
    res_sentiment = "Neutral"
    res_text = ""
    # Check if there were negative comments to make positive resolution more likely/impactful
    was_negative_experience = "Negative" in comment_sentiments or sdesc_sentiment == "Negative" or desc_sentiment == "Negative"

    if random.random() < 0.5 or (was_negative_experience and random.random() < 0.8):
         # More likely to add a positive note if it was a struggle or just randomly
         res_sentiment = "Positive"
         # Add placeholders for cause/number if needed
         res_text = fill_placeholders(random.choice(POSITIVE_PATTERNS))

    else: # Neutral resolution note
         res_sentiment = "Neutral"
         # Add placeholders for cause/number if needed
         res_text = fill_placeholders(random.choice(NEUTRAL_PATTERNS_RES)
                                      .replace("[CAUSE]", fake.bs())
                                      .replace("[NUMBER]", f"{random.randint(1000,9999)}")
                                      )

    current_ticket_entries.append({
        "entry_id": entry_id_counter, "ticket_ref_id": ticket_id, "source_field": "resolution_notes",
        "text_content": res_text, "sentiment": res_sentiment, "resolution_time_minutes": resolution_time
    })
    entry_id_counter += 1

    # Add generated entries for this ticket to the main list
    data.extend(current_ticket_entries)

    # Safety break if generation logic has issues
    if entry_id_counter > NUM_RECORDS_TARGET * 3:
         print("Warning: Safety break triggered. Check generation logic.")
         break


# --- Create DataFrame and Save ---
df = pd.DataFrame(data)

# Ensure we have exactly the target number of records (or slightly more/less is ok)
df = df.head(NUM_RECORDS_TARGET)

print(f"Generated {len(df)} records.")
print("Sample data:")
print(df.head())
print("\nSentiment distribution:")
print(df['sentiment'].value_counts(normalize=True))

df.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"\nSynthetic dataset saved to {OUTPUT_CSV_FILE}")