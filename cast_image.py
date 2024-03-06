import SimpleITK as sitk
import argparse
import numpy as np


def main(args):
        img = sitk.ReadImage(args.img)
        img_np = sitk.GetArrayFromImage(img)

        if args.clip:
                img_np = img_np.clip(args.clip[0], args.clip[1])

        if args.mul:
                img_np = img_np.astype(float)*args.mul

        img_np = img_np.astype(args.type)

        if(args.squeeze):
                img_np = img_np.squeeze()
        
        out_img = sitk.GetImageFromArray(img_np, isVector=args.isVector)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(args.out)
        writer.UseCompressionOn()
        writer.Execute(out_img)



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--img', type=str, help='Input image', required=True)
        parser.add_argument('--clip', type=float, nargs='+', help='Clip the output', default=None)
        parser.add_argument('--mul', type=float, help='Multiply', default=None)
        parser.add_argument('--isVector', type=int, help='Is vector', default=0)
        parser.add_argument('--squeeze', type=int, help='Squeeze', default=0)
        parser.add_argument('--out', type=str, help='Output image', default="out.nrrd")
        parser.add_argument('--type', type=str, help='Output type', default="ubyte")

        args = parser.parse_args()

        main(args)