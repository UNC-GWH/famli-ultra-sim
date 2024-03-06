import os
from tempfile import gettempdir

import numpy as np
import argparse
import SimpleITK as sitk

import subprocess

def main(args):

    if args.img:
        img = sitk.ReadImage(args.img)
        img_size = img.GetSize()

        s = str(1)
        e = str(img_size[-1])
        rl_id = os.path.splitext(os.path.basename(args.img))[0]
    else:
        rl_id = str(args.rl_id)
        s = str(args.s)
        e = str(args.e)

    
    copy_rotation = args.copy_rl + "-" + rl_id + "_R"
    copy_location = args.copy_rl + "-" + rl_id + "_L"

    command = ["blender", "-b", args.blender_fn, "-o", "./ren_tmp/", "--python", args.python_fn, "-s", s, "-e", e, "-a", "--engine", "CYCLES" ,"--", "--probe_fn", args.probe_fn, "--copy_rotation", copy_rotation, "--copy_location", copy_location, "--out", args.out, "--cycles-device", "CUDA"]
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE)

            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Export probe params wrapper. It will execute export probe params using as input the image and setting the -s and -e flags in blender for start and end frames', formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

    parser.add_argument("--blender_fn", type=str, default="/mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/Pregnant_Fetus_Uterus_Blend_2-82/Pregnant_Fetus.blend")
    parser.add_argument("--python_fn", type=str, default="/mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/famli-ultra-sim/export_probe_params.py")
    
    se_group = parser.add_mutually_exclusive_group(required=True)
    se_group.add_argument("--img", type=str, help="Image blind sweep or flyto, the rl_id is extracted from the filename")
    se_group.add_argument("--rl_id", type=str, help="Name of the elements/sweep or rl_id")
    
    parser.add_argument("--s", type=int, help='Start of frames', default=1)
    parser.add_argument("--e", type=int, help="End of frames", default=250)
    

    
    parser.add_argument("--copy_rl", type=str, help="Name of the objects in blender for location and rotation. These are set to the probe and exported during the rendering", required=True)
    parser.add_argument("--probe_fn", type=str, help="Probe filename reference fan or plane", required=True)
    parser.add_argument('--out', type=str, help='Output directory', default='./out')

    args = parser.parse_args()
    
    main(args)