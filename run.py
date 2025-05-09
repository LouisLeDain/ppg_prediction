import argparse
from ppg_pipeline import PPGPipeline

def main(audio_file, sample_rate = 16000, gpu = None,  should_print = False, save_folder = None, visualization_save_folder = None, save = False, visualize = False):
    # Initialize the PPGPipeline with the desired sample rate
    pipeline = PPGPipeline(sample_rate=sample_rate, gpu=gpu)

    # Activate the pipeline
    pipeline.forward(audio_file=audio_file, should_print=should_print, save_folder=save_folder, visualization_save_folder=visualization_save_folder, save=save, visualize=visualize)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Compute PPGs from an audio file.')
    parser.add_argument('--sample_rate', type=int, default = 16000, help='Sample rate of the audio file')
    parser.add_argument('--gpu', type=str, default=None, help='GPU device to use (default: cpu)')
    parser.add_argument('--audio_file', required=True, help='Path to the audio file')
    parser.add_argument('--should_print', help='Print the PPGs sequence')
    parser.add_argument('--save_folder', type=str, help='Path to save the PPGs')
    parser.add_argument('--visualization_save_folder', type=str, help='Path to save the PPGs visualization')
    parser.add_argument('--save', help='Save the PPGs')
    parser.add_argument('--visualize', help='Visualize the PPGs')

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.audio_file, args.sample_rate, args.gpu, args.should_print, args.save_folder, args.visualization_save_folder, args.save, args.visualize)
