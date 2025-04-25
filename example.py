# !pip install transformers

import soundfile as sf
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoProcessor, AutoModelForCTC
from torchinfo import summary

'''
Possible interesting models to use:
- facebook/wav2vec2-base-960h
- facebook/wav2vec2-large-960h
- mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme PAS MAL OUAIS
- slplab/wav2vec2_xlsr50k_english_phoneme PAS MAL OUAIS
- Bluecast/wav2vec2-Phoneme FACILE À LIRE MAIS SOURCES BIZARRES UN PEU
'''


# # load pretrained model Wav2Vec2ForCTC
# processor = Wav2Vec2Processor.from_pretrained("slplab/wav2vec2_xlsr50k_english_phoneme ")
# model = Wav2Vec2ForCTC.from_pretrained("slplab/wav2vec2_xlsr50k_english_phoneme ")

# Load model directly other than facebook

processor = AutoProcessor.from_pretrained("Bluecast/wav2vec2-Phoneme")
model = AutoModelForCTC.from_pretrained("Bluecast/wav2vec2-Phoneme")

# Check model summary

# summary(model, input_size=(1, 16000), col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], row_settings=["var_names"])
print(model)

# Load an example audio file

audio_input, sample_rate = librosa.load('data/HJK/HJK/wav/arctic_a0001.wav', sr=16000)
print("Original sample rate: ", sample_rate)
print("Audio shape: ", audio_input.shape)

# Pad input values and return pt tensor

input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
print("Input values shape: ", input_values.shape)


# INFERENCE

# retrieve logits & take argmax
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe
transcription = processor.decode(predicted_ids[0])
print(transcription)

