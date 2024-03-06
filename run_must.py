import os
from tempfile import gettempdir

import numpy as np
import nrrd
import pickle
from pathlib import Path
import argparse

import subprocess
import scipy.io as sio

def main(args):

    difussor_fn_arr = []

    if args.dir:
        for diffusor_fn in Path(args.dir).glob('*.nrrd'):
            difussor_fn_arr.append(diffusor_fn.as_posix())
    elif args.img:
        difussor_fn_arr.append(args.img)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    for diffusor_fn in difussor_fn_arr:
        out_simu_fn = os.path.join(args.out, os.path.basename(diffusor_fn).replace('.nrrd', '_simu.mat'))        
        
        if not os.path.exists(out_simu_fn):
            command = ["MATLAB", "-batch", "addpath('/home/jprieto/MATLAB Add-Ons/Toolboxes/MUST');addpath('{src}');must_simu('{diffusor_fn}', '{out_simu_fn}')".format(src=os.path.dirname(__file__),diffusor_fn=diffusor_fn, out_simu_fn=out_simu_fn)]

            print(command)
            subprocess.run(command, stdout=subprocess.PIPE) 

        # if os.path.exists(out_simu_fn):
        #     command = ["MATLAB", "-batch", "addpath('/home/jprieto/MATLAB Add-Ons/Toolboxes/MUST');addpath('{src}');must_simu_temp('{diffusor_fn}', '{out_simu_fn}')".format(src=os.path.dirname(__file__),diffusor_fn=diffusor_fn, out_simu_fn=out_simu_fn)]

        #     print(command)
        #     subprocess.run(command, stdout=subprocess.PIPE) 

        if os.path.exists(out_simu_fn):
            out_simu = sio.loadmat(out_simu_fn)
            b_mode_np = out_simu["imageData"]
            if b_mode_np.shape[-1] == 3:
                b_mode_np = b_mode_np[:,:,0]
            # b_mode_np = out_simu["I"]
            # print(b_mode_np.shape)
            # IQ = out_simu["IQ"]
            # RF = out_simu["RF"]

            header = nrrd.read_header(diffusor_fn)
            
            out_b_mode_fn = os.path.join(args.out, os.path.basename(diffusor_fn).replace('.nrrd', '.nrrd'))
            print("Writing:", out_b_mode_fn)
            nrrd.write(out_b_mode_fn, b_mode_np, index_order='C')

            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run MUST simulation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    probe_params_group = parser.add_mutually_exclusive_group(required=True)
    probe_params_group.add_argument('--dir', type=str, help='Input dir with *.nrrd files. Every file is an input to the simulation')
    probe_params_group.add_argument('--img', type=str, help='Input img diffusor')
    parser.add_argument('--out', type=str, help='Output directory', default='./out')

    args = parser.parse_args()
    
    main(args)