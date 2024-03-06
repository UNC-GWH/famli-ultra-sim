import os
import glob
import nrrd
from PIL import Image
import numpy as np
import argparse

def convert_nrrd_to_jpg(nrrd_file_path, jpg_file_path):
    # Read the nrrd file
    data, header = nrrd.read(nrrd_file_path)
    
    # Normalize and convert to 8-bit
    if data.dtype != np.uint8:
        data = (data / np.max(data) * 255).astype(np.uint8)

    # Take the middle slice for 3D images
    if data.ndim == 3:
        slice_index = data.shape[0] // 2
        data = data[slice_index]

    # Convert to PIL Image and save as JPG
    img = Image.fromarray(data)
    img.save(jpg_file_path)

def process_directory(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all .nrrd files in the directory and subdirectories
    for nrrd_file in glob.glob(input_folder + '/**/*.nrrd', recursive=True):
        # Get the name of the parent folder and the nrrd file name
        parent_folder_name = os.path.basename(os.path.dirname(nrrd_file))
        nrrd_file_name = os.path.splitext(os.path.basename(nrrd_file))[0]

        # Construct the corresponding output path
        new_file_name = f"{parent_folder_name}_{nrrd_file_name}.jpg"
        jpg_file_path = os.path.join(output_folder, new_file_name)

        # Convert the file
        convert_nrrd_to_jpg(nrrd_file, jpg_file_path)
        print(f'Converted: {nrrd_file} to {jpg_file_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NRRD files to JPG')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing NRRD files')
    parser.add_argument('output_folder', type=str, help='Path to the output folder for JPG files')
    args = parser.parse_args()

    process_directory(args.input_folder, args.output_folder)
