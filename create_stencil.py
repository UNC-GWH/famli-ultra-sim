import vtk
import os
import numpy as np
import SimpleITK as sitk
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path
import argparse 
    
from multiprocessing import Pool, cpu_count

class BlenderMeshToImage():
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def __call__(self, mesh_fn):
        input_mesh_filename, output_image_filename = mesh_fn

        self.vtk_mesh_to_binary_image(input_mesh_filename, output_image_filename)

    def vtk_mesh_to_binary_image(self, input_mesh_filename, output_image_filename):
        print("Processing:", input_mesh_filename)
        # Read the input mesh
        reader = vtk.vtkSTLReader()
        reader.SetFileName(input_mesh_filename)
        reader.Update()

        surf = reader.GetOutput()
        bounds = np.array(surf.GetBounds())


        dimensions = np.array(self.dimensions)
        spacing = np.abs((bounds[[0,2,4]] - bounds[[1,3,5]]))/dimensions
        
        origin = np.min([bounds[[0,2,4]], bounds[[1,3,5]]], axis=0)

        print(dimensions, spacing, origin)
        # print(dimensions, spacing, origin)

        # Create a binary image with the desired spacing, dimensions, and origin
        white_image = vtk.vtkImageData()
        white_image.SetSpacing(spacing)
        white_image.SetDimensions(dimensions)
        white_image.SetExtent(0, dimensions[0] - 1, 0, dimensions[1] - 1, 0, dimensions[2] - 1)
        white_image.SetOrigin(origin)
        white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # Fill the image with foreground values
        white_value = 1.0
        count = white_image.GetNumberOfPoints()
        for i in range(count):
            white_image.GetPointData().GetScalars().SetTuple1(i, white_value)

        # Create a stencil from the mesh
        poly_data_normals = vtk.vtkPolyDataNormals()
        poly_data_normals.SetInputConnection(reader.GetOutputPort())

        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputConnection(poly_data_normals.GetOutputPort())
        stencil.SetOutputOrigin(origin)
        stencil.SetOutputSpacing(spacing)
        stencil.SetOutputWholeExtent(white_image.GetExtent())

        # Convert the stencil to a binary image
        img_stencil = vtk.vtkImageStencil()
        img_stencil.SetInputData(white_image)
        img_stencil.SetStencilConnection(stencil.GetOutputPort())
        img_stencil.ReverseStencilOff()
        img_stencil.SetBackgroundValue(0)
        img_stencil.Update()

        array = vtk.vtkUnsignedCharArray()
        array.SetNumberOfTuples(np.prod(dimensions))
        array.SetVoidArray(img_stencil.GetOutput().GetScalarPointer(), np.prod(dimensions), 1)

        img_np = vtk_to_numpy(array).reshape(dimensions)


        img = sitk.GetImageFromArray(img_np)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)

        sitk.WriteImage(img, output_image_filename)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create stencil image from mesh in stl format', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dir', type=str, help='Input dir with stl files', required=True)
    parser.add_argument('--dimensions', type=int, help='Output dimension', nargs='+', default=[256, 256, 256])
    # parser.add_argument('--max_size', type=float, help='Max output size', default=512)
    args = parser.parse_args()
    
    # "/mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/Pregnant_Fetus_Uterus_Blend_2-82/stl_export/arms"
    input_dir = args.dir
    mesh_fn = []
    for file in Path(input_dir).rglob('*.stl'):
        
        input_mesh_filename = str(file)   

        output_image_filename = input_mesh_filename.replace('.stl', '.nrrd')

        if not os.path.exists(output_image_filename):
            mesh_fn.append((input_mesh_filename, output_image_filename))

            

    with Pool(cpu_count()) as p:
        p.map(BlenderMeshToImage(args.dimensions), mesh_fn)
