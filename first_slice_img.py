import SimpleITK as sitk
import argparse
import numpy as np


def main(args):
        img = sitk.ReadImage(args.img)
        img_np = sitk.GetArrayFromImage(img)

        img_np = img_np[1:2, :, :]
        
        out_img = sitk.GetImageFromArray(img_np)
        out_img.SetDirection(img.GetDirection())
        out_img.SetOrigin(img.GetOrigin())
        spacing = np.array(img.GetSpacing())
        spacing[2] = spacing[2]*2
        out_img.SetSpacing(spacing)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(args.out)
        writer.UseCompressionOn()
        writer.Execute(out_img)



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--img', type=str, help='Input image', required=True)
        parser.add_argument('--out', type=str, help='Output image', default="out.nrrd")
        

        args = parser.parse_args()

        main(args)