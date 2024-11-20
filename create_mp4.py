import argparse
import cv2
import SimpleITK as sitk
import os
import numpy as np

def read_nrrd_files_from_directory_and_create_video(input_dir, output_filename):
    # Check if the input directory exists
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return

    nrrd_filenames = [f for f in os.listdir(input_dir) if f.endswith('.nrrd')]
    nrrd_filenames = sorted(nrrd_filenames, key=lambda x: int(x.split('.')[0]))    

    if len(nrrd_filenames) == 0:
        print("No .nrrd files found in the directory.")
        return

    # Assume the first file to get the dimensions
    first_frame = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir, nrrd_filenames[0])))  
    print(first_frame.shape)  
    print(first_frame.min(), first_frame.max())
    if len(first_frame.shape) == 2:
        first_frame = np.expand_dims(first_frame, axis=0)
    h, w = first_frame.shape[1:]
    print(h, w)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    frame_per_second = 24  # Define the FPS for the video
    writer = cv2.VideoWriter(output_filename, fourcc, frame_per_second, (w, h))

    for filename in nrrd_filenames:        
        frame = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir, filename)))
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=0)
        # frame = frame*255.0
        # frame = frame.clip(0, 255)
        # frame = frame.astype('uint8')
        frame = frame.transpose((1, 2, 0)).repeat(3, axis=2)  # Convert to HxWxC
        writer.write(frame)

    writer.release()
    print(f"Video saved as {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .nrrd files in a directory to a video.")
    parser.add_argument("--dir", type=str, help="Directory containing .nrrd files.", required=True)
    parser.add_argument("--out", type=str, help="Output video filename.", default="out.mp4")

    args = parser.parse_args()
    read_nrrd_files_from_directory_and_create_video(args.dir, args.out)
