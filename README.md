# Sentiment Analysis Model

A machine learning project for sentiment analysis of text using transformer-based models. This project includes data generation, model training, and a Flask API for inference.

## Project Overview

This project implements a sentiment analysis system that can classify text into three sentiment categories: Positive, Negative, and Neutral. It's particularly designed for analyzing service desk tickets, comments, and descriptions.

### Features

- **Data Generation**: Create synthetic service desk ticket data with sentiment labels
- **Model Training**: Fine-tune transformer models on sentiment data
- **Inference API**: Expose the trained model through a Flask REST API
- **Apple Silicon Support**: Optimized for MPS (Metal Performance Shaders) on Apple Silicon

## Getting Started

### Prerequisites

- Python 3.8+
- macOS with Apple Silicon (for MPS acceleration) or any system with Python support

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd sentiment-analysis-model
   ```

2. Set up the environment:
   ```
   bash install.sh
   ```
   
   This will:
   - Create a Python virtual environment
   - Install PyTorch with MPS support
   - Install required dependencies

## Usage

### Generate Synthetic Data

If you don't have your own labeled sentiment data, you can generate synthetic service desk ticket data:

```
python generate.py
```

This will create a CSV file (`synthetic_servicenow_sentiment_1000.csv`) with 1000 synthetic records.

### Train the Model

Train the sentiment analysis model on your data:

```
python train.py
```

By default, this uses DistilBERT, but you can modify the configuration in `train.py` to use other models like BERT, RoBERTa, or ALBERT.

### Run the API Server

Start the Flask API server to serve predictions:

```
python app.py
```

The server will be available at `http://localhost:5145`.

### Make Predictions

Send a POST request to the API:

```
curl -X POST http://localhost:5145/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Thanks! The app is working perfectly now."}'
```

## Project Structure

- `generate.py`: Script to generate synthetic sentiment data
- `train.py`: Script to train and evaluate the sentiment analysis model
- `app.py`: Flask API for serving predictions
- `install.sh`: Installation script for dependencies
- `requirements.txt`: List of Python dependencies
- `sentiment-finetuned-mps/`: Directory where the trained model is saved (created after training)

## Model Configuration

The default configuration uses DistilBERT for a good balance of performance and speed. You can modify the following parameters in `train.py`:

- `MODEL_CHECKPOINT`: Choose the base model (DistilBERT, BERT, RoBERTa, ALBERT)
- `NUM_EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Learning rate for training
- `TRAIN_BATCH_SIZE`: Batch size for training
- `MAX_TOKEN_LENGTH`: Maximum token length for input text

## Performance Considerations

- The model is optimized for Apple Silicon using MPS acceleration
- For larger models or datasets, consider increasing `GRADIENT_ACCUMULATION_STEPS` and reducing batch sizes
- FP16 precision can be enabled for potential speedup on compatible hardware

## License

[Specify your license here]

## Acknowledgments

- Hugging Face Transformers library
- PyTorch
- Flask
