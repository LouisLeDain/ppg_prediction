import soundfile as sf
import torch
import torch.nn.functional as F
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoProcessor, AutoModelForCTC
from torchinfo import summary
import numpy as np

minimum_length = 400

# Load model and processor

processor = AutoProcessor.from_pretrained("mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme")
model = AutoModelForCTC.from_pretrained("mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme")

# Load audio file

audio_input, sample_rate = librosa.load('/raid/home/automatants/ledain_lou/ppg_prediction/data/HJK/HJK/wav/arctic_a0001.wav', sr=16000)
print('Length of audio input : ', len(audio_input))

# Cut into 50ms sub-recordings

n_frames = int(sample_rate * 0.025)  # 25ms
print(f'Number frames for 10ms sub-recordings : {n_frames}')

audio_input_sub = [audio_input[start:(start+n_frames)] for start in range(0, len(audio_input), n_frames)]
if len(audio_input_sub[-1]) != n_frames :
    audio_input_sub.pop()

print('Number of sub-recordings : ', len(audio_input_sub))
print('Length of last element : ', len(audio_input_sub[-1]))

# Pad input values and return pt tensor

input_values_sub = []

for k in range(len(audio_input_sub)):
    input_values = processor(audio_input_sub[k], sampling_rate=sample_rate, return_tensors="pt").input_values
    input_values_sub.append(input_values)

print(f'Shape of input values : {(len(input_values_sub),input_values_sub[0].shape)}')

# Inference

# Concatenate the tensors along the first dimension
result_tensor = torch.cat(input_values_sub, dim=0)
print(f'Shape of result tensor before padding : {result_tensor.shape}')
padded_tensor = torch.nn.functional.pad(result_tensor, (0, minimum_length - result_tensor.size(1)))
#padded_tensor = result_tensor
print(f'Shape of padded tensor : {padded_tensor.shape}')

logits = model(padded_tensor).logits
print(f'Shape of logits : {logits.shape}')
predicted_ids = torch.argmax(logits, dim=-1)
print(f'Shape of predicted ids : {predicted_ids.shape}')
predicted_sequence = predicted_ids.flatten()
print(f'Shape of predicted sequence : {predicted_sequence.shape}')
print(f'Predicted sequence : {predicted_sequence}')
# transcribe
transcription = processor.decode(predicted_sequence)
print(type(transcription))
print(f'Transcription : {transcription}')

probs = F.softmax(logits, dim=-1)
probs = probs.squeeze(1)
print(f'Shape of probs : {probs.shape}')

print(f'Number of classes : {probs.shape[1]}')
classes = processor.tokenizer.get_vocab()
print(f'Classes : {classes}')

# Show phoneme posterior probabilities

import matplotlib.pyplot as plt
import os

plt.figure(figsize=(20, 4))
plt.imshow(probs.detach().numpy().T, aspect='auto', origin='lower')
plt.colorbar()
plt.title('Phoneme Posterior Probabilities')
plt.xlabel('Time (10ms sub-recordings)')
plt.ylabel('Phoneme classes')
plt.savefig('ppg.png')

