import os
import numpy as np
import SimpleITK as sitk
import argparse
import pandas as pd


def main(args):

    img = sitk.ReadImage(args.img)
    img_np = sitk.GetArrayFromImage(img)

    out_np = np.zeros(img_np.shape, dtype=args.dtype)    

    df = pd.read_csv(args.csv)

    for idx, row in df.iterrows():

        label = row['label']
        print(row)
        
        mean = row[args.mean]
        std = row[args.std]

        # out_np[img_np == label] = (np.random.normal(loc=mean, scale=std, size=img_np.shape))[img_np == label]
        out_np[img_np == label] = np.clip(np.random.normal(loc=mean, scale=std, size=img_np.shape)[img_np == label], a_min=args.a_min, a_max=args.a_max)

    out_img = sitk.GetImageFromArray(out_np)
    out_img.SetSpacing(img.GetSpacing())
    out_img.SetOrigin(img.GetOrigin())
    out_img.SetDirection(img.GetDirection())

    print("Writing:", args.out)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(args.out)
    writer.UseCompressionOn()
    writer.Execute(out_img)
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Relabel an image with mean and standard deviation. img[img==label]=mean+std', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--csv', type=str, help='CSV file with columns label,mean,std', required=True)
    parser.add_argument('--img', type=str, help='Input labeled image', required=True)
    parser.add_argument('--out', type=str, help='Output image name', default='out.nrrd')
    parser.add_argument('--mean', type=str, help='Name of column to replace the labeled image, it represents the mean value', default='mean')
    parser.add_argument('--std', type=str, help='Name of column to replace the labeled image, it represents the standard deviation value', default='std')
    parser.add_argument('--a_min', type=float, help='Clip minimum value', default=None)
    parser.add_argument('--a_max', type=float, help='Clip maximum value', default=None)
    parser.add_argument('--dtype', type=str, help='Output dtype', default='float')
    
    args = parser.parse_args()

    main(args)
