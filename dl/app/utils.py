import SimpleITK as sitk
import numpy as np
from vtk.util import numpy_support
from vtk import vtkImageData, vtkRenderWindow, vtkRenderer, vtkRenderWindowInteractor, vtkPolyDataReader, vtkSTLReader, vtkPolyDataMapper, vtkActor
import os
import cv2 

def readMp4(fname):
    try:        
        cap = cv2.VideoCapture(fname)
        success = cap.isOpened()
        if not success:
            raise IOError("Error opening the MP4 file")
        frames = []
        while success:
            success,frame = cap.read()
            if frame is not None:
                frames.append(frame)
        all_frames = np.stack(frames, axis=0)
    except Exception as e:
        print("Error reading the dicom file as mp4: Error: \n {}".format(e), file=sys. stderr)
        all_frames = None
    return all_frames

def readImage(file_path):
    # Convert SimpleITK image to a NumPy array
    print("Reading image: ", file_path)
    if os.path.splitext(file_path)[1] == ".mp4":
        img_array = readMp4(file_path)
        img_array = np.flip(img_array[:,:,:,0], axis=1)
        vtk_data_array = numpy_support.numpy_to_vtk(num_array=img_array.ravel(), deep=True)

        # Create a VTK image (vtkImageData)
        vtk_image = vtkImageData()
        vtk_image.SetDimensions(list(img_array.shape)[::-1])
        vtk_image.GetPointData().SetScalars(vtk_data_array)

    else:
        sitk_image = sitk.ReadImage(file_path)
        spacing = sitk_image.GetSpacing()
        sitk_array = sitk.GetArrayFromImage(sitk_image)
        
        if (len(sitk_array.shape) == 4):
            sitk_array = sitk_array[:,:,:,0]
            sitk_array = np.flip(sitk_array, axis=1)
        
        # sitk_array = sitk_array.transpose(2,1,0) # Transpose dimensions to match VTK's order

        # Convert NumPy array to VTK array (vtkImageData)
        # print(sitk_array.shape)
        vtk_data_array = numpy_support.numpy_to_vtk(num_array=sitk_array.ravel(), deep=True)

        # Create a VTK image (vtkImageData)
        vtk_image = vtkImageData()
        vtk_image.SetSpacing(sitk_image.GetSpacing())
        vtk_image.SetOrigin(sitk_image.GetOrigin())
        vtk_image.SetDimensions(sitk_image.GetSize())
        vtk_image.GetPointData().SetScalars(vtk_data_array)

    return vtk_image

def createRenderer():
    # Create the second VTK render window and renderer
    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    
    return {
        "renderer": renderer,
        "renderWindow": renderWindow,
        "renderWindowInteractor": renderWindowInteractor
    }

def readSurf(file_path):
    if file_path.endswith(".stl"):
        surf = readStlData(file_path)
    elif file_path.endswith(".vtk"):
        surf = readPolyData(file_path)
    else:
        return
    return surf

def readPolyData(file_path):
    reader = vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def readStlData(file_path):
    reader = vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def createActor(surf=None, return_mapper=False):
    
    mapper = vtkPolyDataMapper()
    if surf is not None:
        mapper.SetInputData(surf)

    actor = vtkActor()
    actor.SetMapper(mapper)

    if return_mapper:
        return actor, mapper

    return actor

def readCreateActor(file_path, return_mapper=False):
    return createActor(readSurf(file_path), return_mapper=return_mapper)

def readAddSurfaceToRenderer(file_path, renderer):
    surf = readSurf(file_path)
    addSurfaceToRenderer(surf, renderer)
    
def addSurfaceToRenderer(surf, renderer):
    actor = createActor(surf)
    
    renderer.AddActor(actor)
    renderer.ResetCamera()