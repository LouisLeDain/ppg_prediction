import argparse
from ppg_pipeline import PPGPipeline

def main(sample_rate = 16000, gpu = None, audio_file, print = False, save_path = None, visualization_save_path = None, save = False, visualize = False):
    # Initialize the PPGPipeline with the desired sample rate
    pipeline = PPGPipeline(sample_rate=sample_rate, gpu=gpu)

    # Activate the pipeline
    pipeline.forward(audio_file=audio_file, print=print, save_path=save_path, visualization_save_path=visualization_save_path, save=save, visualize=visualize)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Compute PPGs from an audio file.')
    parser.add_argument('--sample_rate', type=int, required=True, help='Sample rate of the audio file')
    parser.add_argument('--gpu', type=str, default='cpu', help='GPU device to use (default: cpu)')
    parser.add_argument('--audio_file', type=str, required=True, help='Path to the audio file')
    parser.add_argument('--print', action='store_true', help='Print the PPGs sequence')
    parser.add_argument('--save_path', type=str, help='Path to save the PPGs')
    parser.add_argument('--visualization_save_path', type=str, help='Path to save the PPGs visualization')
    parser.add_argument('--save', action='store_true', help='Save the PPGs')
    parser.add_argument('--visualize', action='store_true', help='Visualize the PPGs')

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.sample_rate, args.gpu, args.audio_file, args.print, args.save_path, args.visualization_save_path, args.save, args.visualize)
