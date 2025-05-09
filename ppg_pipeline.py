'''
Full ppg computation pipeline with saving and visualization tools
This script uses Max Morrison's ppgs library to compute PPGs from audio files.
'''
import ppgs
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


class PPGPipeline():
    def __init__(self, sample_rate, gpu=None):
        self.phones = ppgs.ppgs.phonemes.PHONEMES  
        self.gpu = gpu
        self.sample_rate = sample_rate
        self.audio_file = None
        self.audio = None
        self.ppgs = None
        self.ppgs_array = None
        self.frame_to_phoneme = None
        self.sequence_ids = None
        self.sequence = None

    def load_audio(self, audio_path):
        # Load speech audio at correct sample rate

        base_name = os.path.basename(audio_path)
        audio_name, _ = os.path.splitext(base_name)

        self.audio_file = audio_name
        self.audio = ppgs.load.audio(audio_path)

    def compute_ppgs(self, should_print = False):
        # Infer PPGs
        audio = self.audio
        sample_rate = self.sample_rate
        gpu = self.gpu
        self.ppgs = ppgs.from_audio(audio = audio, sample_rate = sample_rate, gpu = gpu)
        self.ppgs_array = self.ppgs.to(torch.float32).numpy().squeeze(0)
        self.frame_to_phoneme = self.ppgs_array.transpose(1, 0)

        # Infer corresponding sequence

        sequence_ids = self.frame_to_phoneme.argmax(axis=1)
        decode = lambda x: self.phones[x]
        decoded_sequence = [decode(i) for i in sequence_ids]
        
        cleaned_sequence = [decoded_sequence[i] for i in range(1,len(decoded_sequence)-1) if decoded_sequence[i] != decoded_sequence[i-1]]
        self.sequence_ids = sequence_ids
        self.sequence = cleaned_sequence

        if should_print :
            print(f"PPGs sequence: {self.sequence}")
    
    def save_ppgs(self, save_folder, should_print = False):
        # Save PPGs
        save_path = os.path.join(save_folder, f"{self.audio_file}_ppgs.npy")
        np.save(save_path, self.ppgs_array)

        if should_print:
            print(f"PPGs saved to {save_path}")


    def visualize_ppgs(self, save_folder, should_print = False):
        save_path = os.path.join(save_folder, f"{self.audio_file}_ppgs.png")
        
        # Show PPGs
        plt.imshow(self.ppgs_array, aspect='auto', origin='lower')
        plt.title('PPGs')
        plt.xlabel('Time (10ms frames)')
        plt.ylabel('Phones')
        plt.colorbar()
        plt.savefig(save_path)

        if should_print:
            print(f"PPGs visualization saved to {save_path}")

    def forward(self, audio_file, save_folder = None, visualization_save_folder = None, visualize = False, save = False, should_print = False):
        # Check if audio_file is consists of a list of files
        if isinstance(audio_file, list):
            for file in audio_file:
                self.forward(file, save_folder, visualization_save_folder, visualize, save, should_print)
        
        # Load and process audio
        self.load_audio(audio_file)
        self.compute_ppgs(should_print=should_print)
        if save:
            if save_folder is None:
                print("Please provide a save folder")
            else: 
                self.save_ppgs(save_folder, should_print=should_print)
        if visualize:
            if visualization_save_folder is None:
                print("Please provide a visualization save folder")
            else:
                self.visualize_ppgs(visualization_save_folder, should_print=should_print)