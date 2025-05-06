import ppgs
import torch
import numpy as np

phones = ppgs.ppgs.phonemes.PHONEMES 
print(f"PPGs phonemes: {phones}")

# Load speech audio at correct sample rate
audio_file = 'data/HJK/HJK/wav/arctic_a0001.wav' 
audio = ppgs.load.audio(audio_file)

# Choose a gpu index to use for inference. Set to None to use cpu.
gpu = None 

# Infer PPGs
ppgs = ppgs.from_audio(audio, ppgs.SAMPLE_RATE, gpu=gpu)
print(f"PPG type: {type(ppgs)}")

ppgs_32 = ppgs.to(torch.float32)
ppgs_array = ppgs_32.cpu().numpy().squeeze(0)

frame_to_phoneme = ppgs_array.transpose(1, 0)
sequence_ids = [frame_to_phoneme[i].argmax() for i in range(frame_to_phoneme.shape[0])]
decoded_sequence = [phones[i] for i in sequence_ids]
print(f"Decoded sequence: {decoded_sequence}")

cleaned_sequence = [decoded_sequence[i] for i in range(1,len(decoded_sequence)-1) if decoded_sequence[i] != decoded_sequence[i-1]]
print(f"Cleaned sequence: {cleaned_sequence}")

print(f"PPGs shape: {ppgs_array.shape}")

# Show and save ppg

import matplotlib.pyplot as plt
import os

# Show PPGs
plt.imshow(ppgs_array, aspect='auto', origin='lower')
plt.title('PPGs')
plt.xlabel('Time (frames)')
plt.ylabel('PPG dimension')
plt.colorbar()
plt.savefig('ppgs.png')