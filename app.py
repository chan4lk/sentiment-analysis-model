from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch # Or import tensorflow if you used TF

# --- Configuration ---
MODEL_DIR = "./sentiment-finetuned-mps" # Path to your saved HF model
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" # Use MPS if available

# --- Load Model and Tokenizer (Load once on startup) ---
try:
    print(f"Loading model from {MODEL_DIR} onto device {DEVICE}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.to(DEVICE) # Move model to MPS or CPU
    print("Model loaded successfully.")

    # Create a pipeline for easier inference
    # task='sentiment-analysis' automatically handles tokenization, prediction, label mapping
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE # Use 0 for CUDA if you had a GPU, -1 for CPU, or device object
    )
    print("Pipeline created.")

except Exception as e:
    print(f"Error loading model or creating pipeline: {e}")
    sentiment_pipeline = None # Indicate failure

# --- Create Flask App ---
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if not sentiment_pipeline:
        return jsonify({"error": "Model not loaded"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "Missing 'text' field in JSON payload"}), 400

    try:
        # Perform prediction using the pipeline
        results = sentiment_pipeline(text, return_all_scores=True) # Get scores for all classes

        # The pipeline might return a list if input is a list, handle single text input
        if isinstance(results, list) and len(results) > 0:
             processed_result = results[0] # Take the first result for single text input
        else:
             processed_result = results # Assume it's already the dict

        # Find the label with the highest score if needed (pipeline often does this)
        # Example of extracting label and score:
        # predicted_label = max(processed_result, key=lambda x: x['score'])['label']
        # highest_score = max(processed_result, key=lambda x: x['score'])['score']

        return jsonify(processed_result) # Return the raw pipeline output (list of dicts with labels/scores)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5145) # Run on local network, port 5000