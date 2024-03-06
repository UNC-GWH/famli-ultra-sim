import os
import numpy as np
import SimpleITK as sitk
import argparse
import pandas as pd


def main(args):
    df = pd.read_csv(args.csv)

    df = df.query('study_id == "{study_id}"'.format(study_id=args.study_id))

    tags_df = pd.read_csv(args.tags)

    out_np = np.zeros(args.out_size, dtype=np.ushort)
    out_img = sitk.GetImageFromArray(out_np)
    out_img.SetSpacing(args.output_spacing)
    out_img.SetOrigin(args.out_origin)

    

    
    InterpolatorType = sitk.sitkNearestNeighbor
    resampleImageFilter = sitk.ResampleImageFilter()
    resampleImageFilter.SetInterpolator(InterpolatorType)
    resampleImageFilter.SetReferenceImage(out_img)
    resampled_img = resampleImageFilter.Execute(img)

    resampled_img_np = sitk.GetArrayFromImage(resampled_img).astype(np.ushort)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Merge ALL blind sweeps into a single volume', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--csv', type=str, help='CSV file with columns study_id,tag', required=True)
    parser.add_argument('--study_id', type=str, help='study_id in the csv to generate', required=True)
    parser.add_argument('--tags', type=str, help='CSV with description of the tags to use with start and end position', required=True)
    parser.add_argument('--mount_dir', type=str, help='Mount directory', default="./")    
    parser.add_argument('--out_size', type=int, help='Output size', nargs='+', default=[256, 256, 256])
    parser.add_argument('--out_spacing', type=int, help='Output spacing', nargs='+', default=[0.25, 0.25, 0.25])
    parser.add_argument('--out_origin', type=int, help='Output origin', nargs='+', default=[0.0, 0.0, 0.0])
    
    args = parser.parse_args()

    main(args)
