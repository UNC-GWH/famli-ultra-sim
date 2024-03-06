import os
from tempfile import gettempdir

import numpy as np
import SimpleITK as sitk
import pickle
import glob
import argparse
import scipy.io as sio

import subprocess


def read_probe_params(probe_params_fn):
    return pickle.load(open(probe_params_fn, 'rb'))

def sample_image(probe_params, img, interpolator=None, identity_direction=False):

    probe_origin = probe_params['probe_origin']
    probe_direction = probe_params['probe_direction']
    ref_size = probe_params['ref_size']
    ref_origin = probe_params['ref_origin']
    ref_spacing = probe_params['ref_spacing']
    
    ref = sitk.Image(int(ref_size[0]), int(ref_size[1]), int(ref_size[2]), sitk.sitkFloat32)
    ref.SetOrigin(ref_origin)
    ref.SetSpacing(ref_spacing)
    ref.SetDirection(probe_direction.flatten().tolist())

    resampler = sitk.ResampleImageFilter()
    if interpolator:
        resampler.SetInterpolator(interpolator)
    resampler.SetReferenceImage(ref)

    sample = resampler.Execute(img)
    if identity_direction:
        sample_np = sitk.GetArrayFromImage(sample).squeeze()
        sample_np = np.flip(np.rot90(sample_np, k=1, axes=(0, 1)), axis=0)
        sample = sitk.GetImageFromArray(sample_np)
        sample.SetSpacing(ref_spacing)
    return sample



def main(args):

    img = sitk.ReadImage(args.img)

    sound_speed_img = None
    density_img = None

    if args.run_simulation:
        sound_speed_img = sitk.ReadImage(args.sound_speed)
        density_img = sitk.ReadImage(args.density)
    
    probe_params_fn_arr = []

    if args.probe_params_dir:
        for probe_params_fn in glob.glob(os.path.join(args.probe_params_dir, '*_probe_params.pickle')):
            probe_params_fn_arr.append(probe_params_fn)
        if args.sort:
            fn_arr = [int(os.path.basename(fn).replace('_probe_params.pickle', '')) for fn in probe_params_fn_arr]
            fn_arr_sorted_idx = sorted(range(len(fn_arr)), key=lambda idx: fn_arr[idx])
            probe_params_fn_arr = [probe_params_fn_arr[idx] for idx in fn_arr_sorted_idx]

    elif args.probe_param:
        probe_params_fn_arr.append(args.probe_param)
        
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    mask_img = None
    if args.mask:
        mask_img = sitk.ReadImage(args.mask)

    for probe_params_fn in probe_params_fn_arr:
        probe_params = read_probe_params(probe_params_fn)

        img_map = sample_image(probe_params, img, interpolator=sitk.sitkNearestNeighbor)

        if mask_img is not None:
            mask_img.SetDirection(img_map.GetDirection())
            mask_img.SetSpacing(img_map.GetSpacing())
            mask_img.SetOrigin(img_map.GetOrigin())

            img_map = sitk.Mask(img_map, mask_img)
            

        out_fn = os.path.join(args.out, os.path.basename(probe_params_fn).replace('_probe_params.pickle', '.nrrd'))
        print("Writing:", out_fn)
        sitk.WriteImage(img_map, out_fn)

        if args.run_simulation:

            sound_speed_map = sample_image(probe_params, sound_speed_img)
            out_sound_speed_map_fn = os.path.join(args.out, os.path.basename(probe_params_fn).replace('_probe_params.pickle', '_sound_speed_map.nrrd'))
            sitk.WriteImage(sound_speed_map, out_sound_speed_map_fn)
            # sound_speed_map -= np.min(sound_speed_map)
            # sound_speed_map /= np.max(sound_speed_map)
            # sound_speed_map *= (1601.5913243064965 - 1400.0)
            # sound_speed_map += (1400.0)
            
            density_map = sample_image(probe_params, density_img)
            out_density_map_fn = os.path.join(args.out, os.path.basename(probe_params_fn).replace('_probe_params.pickle', '_density_map.nrrd'))
            sitk.WriteImage(density_map, out_density_map_fn)
            # density_map -= np.min(density_map)
            # density_map /= np.max(density_map)
            # density_map *= (1066.6666666666667 - 933.3333333333334)
            # density_map += (933.3333333333334)

            out_simu_fn = os.path.join(args.out, os.path.basename(probe_params_fn).replace('_probe_params.pickle', '_simu.mat'))

            if not os.path.exists(out_simu_fn):
                command = ["MATLAB", "-batch", "addpath('{src}');us_bmode_phased_array('{out_sound_speed_map_fn}', '{out_density_map_fn}', '{out_simu_fn}')".format(src=os.path.dirname(__file__),out_sound_speed_map_fn=out_sound_speed_map_fn, out_density_map_fn=out_density_map_fn, out_simu_fn=out_simu_fn)]
            
            b_mode_fund_dir = os.path.join(args.out, 'bmode_fund')
            if not os.path.exists(b_mode_fund_dir) or not os.path.isdir(b_mode_fund_dir):
                os.makedirs(b_mode_fund_dir)

            b_mode_harm_dir = os.path.join(args.out, 'bmode_harm')
            if not os.path.exists(b_mode_harm_dir) or not os.path.isdir(b_mode_harm_dir):
                os.makedirs(b_mode_harm_dir)

            if os.path.exists(out_simu_fn):
                out_simu = sio.loadmat(out_simu_fn)
                b_mode_fund = out_simu["b_mode_fund"]
                b_mode_harm = out_simu["b_mode_harm"]

                b_mode_fund_fn = os.path.join(b_mode_fund_dir, os.path.basename(probe_params_fn).replace('_probe_params.pickle', '_bmode_fund.nrrd'))

                b_mode_fund = sitk.GetImageFromArray(b_mode_fund)
                print("Writing:", b_mode_fund_fn)
                sitk.WriteImage(b_mode_fund, b_mode_fund_fn)

                b_mode_harm_fn = os.path.join(b_mode_harm_dir, os.path.basename(probe_params_fn).replace('_probe_params.pickle', '_bmode_harm.nrrd'))
                b_mode_harm = sitk.GetImageFromArray(b_mode_harm)
                print("Writing:", b_mode_harm_fn)
                sitk.WriteImage(b_mode_harm, b_mode_harm_fn)
  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Based on probe parameters extracted from blender, extracts a chuck from the medium that can be used in the US simulation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    probe_params_group = parser.add_mutually_exclusive_group(required=True)
    probe_params_group.add_argument('--probe_params_dir', type=str, help='Input dir with *_probe_params.pickle files')
    probe_params_group.add_argument('--probe_param', type=str, help='Input _probe_params.pickle file')
    parser.add_argument('--img', type=str, help='Image or medium to be sampled from', required=True)
    parser.add_argument('--mask', type=str, help='Mask image', default=None)
    parser.add_argument('--sound_speed', type=str, help='Sound speed image', default=None)
    parser.add_argument('--density', type=str, help='Density image', default=None)
    parser.add_argument('--run_simulation', type=str, help='Run the simulation', default=0)
    parser.add_argument('--out', type=str, help='Output directory', default='./out')
    parser.add_argument('--sort', type=int, help='Sort the filenames', default=1)

    
    args = parser.parse_args()
    
    main(args)