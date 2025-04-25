import zipfile
import os

def unzip_file(zip_file_path, extract_to_directory):
    # Check if the ZIP file exists
    if not os.path.exists(zip_file_path):
        print(f"The file {zip_file_path} does not exist.")
        return

    # Open the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the specified directory
        zip_ref.extractall(extract_to_directory)
        print(f"Files extracted to {extract_to_directory}")

# Example usage
zip_file_path = 'data/HJK.zip'
extract_to_directory = 'data/HJK'
unzip_file(zip_file_path, extract_to_directory)
