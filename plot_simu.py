import os
from tempfile import gettempdir

import SimpleITK as sitk
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scipy.io as sio

def main(args):

    pml_x_size = 10
    pml_y_size = 10
    pml_z_size = 10
    
    sc = 1
    Nx = 320/sc - 2*pml_x_size
    Ny = 320/sc - 2*pml_y_size
    Nz = 320/sc - 2*pml_z_size

    x = 50e-3

    dx = x / Nx
    dy = dx
    dz = dx

    x_axis = [0, Nx * dx * 1e3]
    y_axis = [0, Ny * dy * 1e3]

    simu = sio.loadmat(args.simu)
    sound_speed = sitk.GetArrayFromImage(sitk.ReadImage(args.simu.replace('_simu.mat', '_sound_speed_map.nrrd')))

    scan_lines = simu['scan_lines']
    scan_lines_fund = simu['scan_lines_fund']
    b_mode_fund = simu['b_mode_fund']
    b_mode_harm = simu['b_mode_harm']

    steering_angles = range(-32,33,2)

    # plot the data before and after scan conversion
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(scan_lines, extent=[min(steering_angles), max(steering_angles), min(x_axis), max(x_axis)], aspect='auto')
    axs[0].set_xlabel('Steering angle [deg]')
    axs[0].set_ylabel('Depth [mm]')
    axs[0].set_title('Raw Scan-Line Data')

    axs[1].imshow(scan_lines_fund, extent=[min(steering_angles), max(steering_angles), min(x_axis), max(x_axis)], aspect='auto')
    axs[1].set_xlabel('Steering angle [deg]')
    axs[1].set_ylabel('Depth [mm]')
    axs[1].set_title('Processed Scan-Line Data')

    axs[2].imshow(b_mode_fund, extent=[min(y_axis), max(y_axis), min(x_axis), max(x_axis)], cmap='gray', aspect='auto')
    axs[2].set_xlabel('Horizontal Position [mm]')
    axs[2].set_ylabel('Depth [mm]')
    axs[2].set_title('B-Mode Image')

    plt.tight_layout()
    plt.show()


    # plot the medium and the B-mode images
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(sound_speed[int(sound_speed.shape[0]/2), :, :], extent=[min(y_axis), max(y_axis), min(x_axis), max(x_axis)], aspect='auto')
    axs[0].set_xlabel('Horizontal Position [mm]')
    axs[0].set_ylabel('Depth [mm]')
    axs[0].set_title('Scattering Phantom')

    axs[1].imshow(b_mode_fund, extent=[min(y_axis), max(y_axis), min(x_axis), max(x_axis)], cmap='gray', aspect='auto')
    axs[1].set_xlabel('Horizontal Position [mm]')
    axs[1].set_ylabel('Depth [mm]')
    axs[1].set_title('B-Mode Image')

    axs[2].imshow(b_mode_harm, extent=[min(y_axis), max(y_axis), min(x_axis), max(x_axis)], cmap='gray', aspect='auto')
    axs[2].set_xlabel('Horizontal Position [mm]')
    axs[2].set_ylabel('Depth [mm]')
    axs[2].set_title('Harmonic Image')

    plt.tight_layout()
    plt.show()

   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save bmode', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--simu', type=str, help='Simulation output', required=True)
    # parser.add_argument('--sound_speed', type=str, help='Sound speed', required=True)
    
    args = parser.parse_args()
    
    main(args)