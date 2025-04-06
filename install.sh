python3 -m venv env
source env/bin/activate
# Install PyTorch for Mac (usually auto-detects ARM/MPS)

pip install torch torchvision torchaudio

# Install other libraries (note: bitsandbytes is removed)
pip install transformers[torch] datasets pandas scikit-learn accelerate