import vtk
import os
from tempfile import gettempdir

import numpy as np
import SimpleITK as sitk
import pickle
import glob
import argparse
import scipy.io as sio
import pandas as pd

def read_probe_params(probe_params_fn):
    return pickle.load(open(probe_params_fn, 'rb'))

def sample_image(probe_params, img, interpolator=None):

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

    return resampler.Execute(img)



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
    elif args.probe_param:
        probe_params_fn_arr.append(args.probe_param)
        
    if not os.path.exists(args.out):
        os.makedirs(args.out)


    df = pd.read_csv(args.csv)

    appendPoly = vtk.AppendPolyData()


    for idx, row in df.iterrows():
        
    # Read the mesh data
        reader = vtk.vtkSTLReader()
        reader.SetFileName(row['surf'])
        reader.Update()

        appendPoly.AddInputData(reader.GetOutput())

    for probe_params_fn in probe_params_fn_arr:
        probe_params = read_probe_params(probe_params_fn)

        # Define the plane for intersection
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, 0)  # Set the origin of the plane
        plane.SetNormal(0, 0, 1)  # Set the normal of the plane

        # Use vtkCutter to intersect the mesh with the plane
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputConnection(reader.GetOutputPort())
        cutter.Update()

        # Map the contours to a color
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cutter.GetOutputPort())

    # Create an actor to visualize the contours
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a rendering window, renderer, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Background color

    # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()

  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cut mesh using probe parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    probe_params_group = parser.add_mutually_exclusive_group(required=True)
    probe_params_group.add_argument('--probe_params_dir', type=str, help='Input dir with *_probe_params.pickle files')
    probe_params_group.add_argument('--probe_param', type=str, help='Input _probe_params.pickle file')
    parser.add_argument('--csv', type=str, help='CSV file with column surf', required=True)
    parser.add_argument('--out', type=str, help='Output directory', default='./out')

    
    args = parser.parse_args()
    
    main(args)
