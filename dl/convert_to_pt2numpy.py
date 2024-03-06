import argparse
import torch
import nrrd
import os

# python tensor_to_nrrd.py path/to/your_tensor.pt output_path/output.nrrd

def tensor_to_nrrd(tensor_path, nrrd_path):
    # Load the tensor
    tensor = torch.load(tensor_path)
    
    # Convert the tensor to a numpy array
    numpy_array = tensor.cpu().squeeze().numpy()
    
    # Save the numpy array as an NRRD file
    print(tensor.shape, numpy_array.shape)
    print(nrrd_path)
    nrrd.write(nrrd_path, numpy_array, index_order='C')

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Convert a tensor to NRRD format.")
    parser.add_argument("tensor_path", type=str, help="Path to the tensor file.")    
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Convert and save the tensor as an NRRD file
    print(args.tensor_path)

    tensor_to_nrrd(args.tensor_path, args.tensor_path.replace(os.path.splitext(args.tensor_path)[1], '.nrrd'))

if __name__ == "__main__":
    main()

