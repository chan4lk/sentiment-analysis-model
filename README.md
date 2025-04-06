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

```bash
python generate.py
```


## MLX Train

```bash
pip install mlx_lm
```

```bash
mlx_lm.lora --model microsoft/Phi-3-mini-4k-instruct --train --data ./data --iters 1000
```

## MLX Evaluate

```bash
mlx_lm.generate --model microsoft/Phi-3-mini-4k-instruct --adapter-path ./adapters --max-token 2048 --prompt "<|user|>\nClassify the sentiment (Positive, Negative, or Neutral) for this ServiceNow ticket text:\nRequesting access to Outlook for new user Marc Crawford.<|end|>"
```


## without adaptors
```bash
mlx_lm.generate --model microsoft/Phi-3-mini-4k-instruct --max-token 2048 --prompt "<|user|>\nClassify the sentiment (Positive, Negative, or Neutral) for this ServiceNow ticket text:\nRequesting access to Outlook for new user Marc Crawford.<|end|>"
```

## merge model
```bash
mlx_lm.fuse --model microsoft/Phi-3-mini-4k-instruct
```


## Convert to gguf

```bash
source venv/bin/activate
python convert_hf_to_gguf.py \
  /Users/chandima/repos/sentiment-analysis-model/fused_model/ \
  --outfile /Users/chandima/repos/sentiment-analysis-model/gguf/sentiment-finetuned-mps.gguf \
  --outtype f16 # Or f32, q8_0, q4_k_m etc.
```

## Create ollama model
```bash
ollama create phi3ft -f Modelfile
```

## Run ollama model
```bash
ollama run phi3ft "<|user|>\nClassify the sentiment (Positive, Negative, or Neutral) for this ServiceNow ticket text:\nRequesting access to Outlook for new user Marc Crawford.<|end|>" 
```