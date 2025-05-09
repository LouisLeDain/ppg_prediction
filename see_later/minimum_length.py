# Finding minimum input size for model

import soundfile as sf
import torch
import torch.nn.functional as F
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoProcessor, AutoModelForCTC
from torchinfo import summary
import numpy as np

# Load model and processor

processor = AutoProcessor.from_pretrained("mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme")
model = AutoModelForCTC.from_pretrained("mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme")

for i in range(1, 600):
    dummy_tensor = torch.randn(1, i)
    print(f"Dummy tensor shape: {dummy_tensor.shape}")
    try:
        logits = model(dummy_tensor).logits
        print(f'Shape of logits: {logits.shape}')
    except RuntimeError as e:
        print(f"Error computing logits: {e}")

# Minimum length for model : 400