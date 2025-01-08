import nrrd
import pandas as pd
import numpy as np

def apply_label_mapping(img, df, args):
    # Read the nrrd file
    # Convert the DataFrame to a dictionary for faster access
    labels = np.array(df[args.target].values)
    return labels[img]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='Path to the nrrd file')
    parser.add_argument('--csv', type=str, help='CSV files with columns label and label_10')
    parser.add_argument('--target', type=str, help='label target column', default='label_11')
    parser.add_argument('--out', type=str, help='Path to the output file')

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    
    img, header = nrrd.read(args.img)
    
    out_img = apply_label_mapping(img, df, args)

    print("Writing: ", args.out)
    nrrd.write(args.out, out_img, header)

    # Save the modified image
    # nrrd.write(args.out, out_img, header)



