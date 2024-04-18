
import vtk
from vtk.util import numpy_support
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindowInteractor
)

from vtk import (
    vtkResliceImageViewer,
    vtkTransform, 
    vtkTransformPolyDataFilter,
    vtkAxesActor
)

import SimpleITK as sitk
import numpy as np
import pandas as pd


from utils import *
import os
import glob 

class ViewComponents:
    def __init__(self):
        
        self.image_data = None
        self.mount_point = "/Volumes/med/GWH/Groups/FAMLI/Shared/C1_ML_Analysis/"
        # self.image_data = readImage("/Users/juan/Desktop/M.dcm")
        # image_data = readImage("/Users/juan/Desktop/a622d9a6-da7f-4975-af79-77db76f69792.nrrd")
        # image_data = readImage("c:/Users/juan/Desktop/M.dcm")
        
        # # Create the first VTK render window and renderer
        renderWindowInteractor = vtkRenderWindowInteractor()
        self.resliceViewer = vtkResliceImageViewer()
        self.resliceViewer.SetupInteractor(renderWindowInteractor)
        self.resliceViewer.GetRenderWindow().SetOffScreenRendering(1)

        self.renderer_dict = {}

        self.renderer_dict["main"] = createRenderer()

        axes_actor = vtkAxesActor()
        axes_actor.AxisLabelsOn()
        axes_actor.SetXAxisLabelText("X")
        axes_actor.SetYAxisLabelText("Y")
        axes_actor.SetZAxisLabelText("Z")

        axes_transform = vtkTransform()
        # axes_transform.Translate(-1.0, -1.0, -1.0)
        axes_transform.Scale(0.1, 0.1, 0.1)
        axes_actor.SetUserMatrix(axes_transform.GetMatrix())

        axes_actor.SetPosition(-1, -1, -1)

        self.renderer_dict["main"]["renderer"].AddActor(axes_actor)

        self.df_fn = os.path.join(self.mount_point, "famli_ml_lists/AnalysisLists/Juan/C_dataset_analysis_protocoltagsonly_gaboe230_ge_iq_train.csv")
        self.df = pd.read_csv(self.df_fn)


        ext = os.path.splitext(self.df_fn)[1]
        self.studies_fn = self.df_fn.replace(ext, "_studies" + ext)

        if not os.path.exists(self.studies_fn):
            self.df_studies = self.df[["study_id"]].drop_duplicates()
            self.df_studies.set_index("study_id", inplace=True)
            self.df_studies["tx"] = 0.0
            self.df_studies["ty"] = 0.0
            self.df_studies["tz"] = 0.0
            self.df_studies["rx"] = 0.0
            self.df_studies["ry"] = 0.0
            self.df_studies["rz"] = 0.0
            self.df_studies["completed"] = 0
        else:
            self.df_studies = pd.read_csv(self.studies_fn, index_col="study_id")
        # print(self.df_studies)
        self.study_ids = self.df["study_id"].drop_duplicates().tolist()
        
        self.current_study_idx = -1
        self.study_id = ""
        self.study_images = {}


        mount_point_simulated = os.path.join(self.mount_point, "src/blender/simulated_data_export/")
        model_fn = os.path.join(mount_point_simulated, "breech_0/fetus/Fetus_Model.stl")

        self.main_actor, self.main_mapper = readCreateActor(model_fn, return_mapper=True)

        self.renderer_dict["main"]["renderer"].AddActor(self.main_actor)

        self.template_arr = ["src/diffusion-models/blender/studies/FAM-025-0499-5/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-1336-3/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-1398-3/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-0447-5/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-0615-3/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-1453-3/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-0626-2/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-0664-4/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-0749-4/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-1485-1/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-1489-1/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-0754-2/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-1491-2/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-0941-1/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-0795-1/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-0869-1/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-1275-1/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-0950-4/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/studies/FAM-025-1144-4/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_flexed_arms_cross_legs_uncross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_overflexed_arms_cross_legs_uncross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_neutral_arms_cross_legs_uncross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_neutral_arms_cross_legs_cross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_flexed_arms_cross_legs_cross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_overflexed_arms_cross_legs_cross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_neutral_arms_uncross_legs_cross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_flexed_arms_uncross_legs_cross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_overflexed_arms_uncross_legs_cross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_neutral_arms_uncross_legs_uncross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_flexed_arms_uncross_legs_uncross/fetus/Fetus_Model.stl",
            "src/diffusion-models/blender/generic2/head_overflexed_arms_uncross_legs_uncross/fetus/Fetus_Model.stl"]
        
        self.template_surf = []
        self.renderer_dict["bottom"] = []
        self.current_actor_index = 0

        for i in range(5):
            ren_d = createRenderer()
            actor, mapper = createActor(surf=None, return_mapper=True)

            ren_d["mapper"] = mapper
            ren_d["renderer"].AddActor(actor)

            self.renderer_dict["bottom"].append(ren_d)

        for template_fn in self.template_arr:
            surf = readSurf(os.path.join(self.mount_point, template_fn))

            # actor = createActor(surf)

            self.template_surf.append(surf)
            # self.template_actors.append(actor)
            # self.renderer_dict["bottom"].append(createRenderer())
            # self.renderer_dict["bottom"][-1]["renderer"].AddActor(actor)
        
        self.NextTemplateActors()

        self.grid_sweeps = {}
        model_grid_dir = os.path.join(mount_point_simulated, "breech_0/paths")
        for sweep_fn in glob.glob(os.path.join(model_grid_dir, '*.vtk')):
            key = os.path.basename(sweep_fn).split(".")[0].replace("_path", "")
            self.grid_sweeps[key] = readSurf(sweep_fn)

            actor = createActor(self.grid_sweeps[key])
            self.renderer_dict["main"]["renderer"].AddActor(actor)

        self.sweep_actors = []
        model_sweep_dir = os.path.join(mount_point_simulated, "breech_0/tube")
        for sweep_fn in glob.glob(os.path.join(model_sweep_dir, '*.stl')):
            self.sweep_actors.append(readCreateActor(sweep_fn))
        for actor in self.sweep_actors:
            actor.GetProperty().SetOpacity(0.5)
            self.renderer_dict["main"]["renderer"].AddActor(actor)

        ultrasound_fan_fn = os.path.join(mount_point_simulated, "breech_0/probe/ultrasound_fan_2d_iq.stl")

        self.ultrasound_fan_actor = readCreateActor(ultrasound_fan_fn)
        self.ultrasound_fan_actor_transform = vtkTransform()

        self.renderer_dict["main"]["renderer"].AddActor(self.ultrasound_fan_actor)

        self.main_transform = vtk.vtkTransform()
        

    def Interpolate_coordinates(self, tau, sweep_key):
        sweep = self.grid_sweeps[sweep_key]
        sweep_points = sweep.GetPoints()

        idx_c = tau * sweep_points.GetNumberOfPoints()
        idx = int(idx_c)
        delta_idx = idx_c - idx

        if idx == sweep_points.GetNumberOfPoints() - 1:
            p0 = np.array(sweep_points.GetPoint(idx - 1))
            p1 = np.array(sweep_points.GetPoint(idx))
            delta_idx = 1.0
        else:
            p0 = np.array(sweep_points.GetPoint(idx))
            p1 = np.array(sweep_points.GetPoint(idx + 1))
            
        return p0 + delta_idx * (p1 - p0), p1 - p0
        
    def ComputeRotationParams(self, normal, direction):
        normal = normal / np.linalg.norm(normal)
        direction = direction / np.linalg.norm(direction)
    
        # Calculate the rotation axis by taking the cross product of the two vectors
        rotationVector = np.cross(normal, direction)
        # Calculate rotation angle using the dot product
        rotationAngle = np.arccos(np.dot(normal, direction))*180/np.pi

        return rotationAngle, rotationVector
    def OnMouseWheel(self):
        self.UpdateFan()

    def UpdateFan(self):
        slice_num = self.resliceViewer.GetSlice()

        if self.image_data and not self.tag == "":
            tau = slice_num/self.image_data.GetDimensions()[-1]
            
            slice_coordinates, direction = self.Interpolate_coordinates(tau, self.tag)

            self.ultrasound_fan_actor_transform.Identity()
            
            self.ultrasound_fan_actor_transform.Translate(slice_coordinates[0], slice_coordinates[1], slice_coordinates[2])

            if self.tag in ["R1", "R0", "M", "L0", "L1"]:
                rotationAngle, rotationVector = self.ComputeRotationParams(np.array([0.0, 0.0, 1.0]), direction)
                self.ultrasound_fan_actor_transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2])
                self.ultrasound_fan_actor_transform.RotateWXYZ(90, 0.0, 0.0, 1.0)
            else:
                rotationAngle, rotationVector = self.ComputeRotationParams(np.array([0.0, 0.0, 1.0]), -direction)
                self.ultrasound_fan_actor_transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2])

            self.ultrasound_fan_actor.SetUserMatrix(self.ultrasound_fan_actor_transform.GetMatrix())

    def OnTransformSliderChange(self, translation, rotation_angles):

        # Reset the transform to identity to apply new transformations
        self.main_transform.Identity()

        # Apply translation
        self.main_transform.Translate(*translation)
        
        # Apply rotations. VTK applies rotations in the order of Z, Y, X by default
        self.main_transform.RotateX(rotation_angles[0])
        self.main_transform.RotateY(rotation_angles[1])
        self.main_transform.RotateZ(rotation_angles[2])

        self.main_actor.SetUserMatrix(self.main_transform.GetMatrix())

    def LoadStudy(self, idx, study_id):
        self.current_study_idx = idx
        self.study_id = study_id
        self.study_images = {}

        for idx, row in self.df.query(f'study_id == "{study_id}"')[['tag', 'file_path']].iterrows():
            self.study_images[row['tag']] = {"text": row['tag'], "file_path": row['file_path']}

        return [v for _, v in self.study_images.items()]
    
    def LoadImg(self, tag):
        self.tag = tag
        if tag in self.study_images:
            img_path = self.study_images[tag]["file_path"]

            img_path = os.path.join(self.mount_point, img_path)
            self.image_data = readImage(img_path)
            self.resliceViewer.SetInputData(self.image_data)

            self.UpdateFan()

    def GetStudyTransform(self, study_id):
        print(self.df_studies.query(f'study_id == "{study_id}"')[["tx", "ty", "tz", "rx", "ry", "rz"]])
        print(self.df_studies.loc[study_id, ["tx", "ty", "tz", "rx", "ry", "rz"]].values)
        if study_id in self.df_studies.index:
            return self.df_studies.loc[study_id, ["tx", "ty", "tz", "rx", "ry", "rz"]].values
        return [0, 0, 0, 0, 0, 0]

    def Save(self, tx, ty, tz, rx, ry, rz):
        # Check if the specified ID exists in the DataFrame
        if self.study_id in self.df_studies.index:
            # Update the row columns as specified
            self.df_studies.at[self.study_id, "tx"] = tx
            self.df_studies.at[self.study_id, "ty"] = ty
            self.df_studies.at[self.study_id, "tz"] = tz
            self.df_studies.at[self.study_id, "rx"] = rx
            self.df_studies.at[self.study_id, "ry"] = ry
            self.df_studies.at[self.study_id, "rz"] = rz
            self.df_studies.at[self.study_id, "completed"] = 1
            # Write the updated DataFrame back to the file
            self.df_studies.to_csv(self.studies_fn)
    
    def GetStudyButtonColor(self, study_id):
        if study_id in self.df_studies.index:
            return "#00E676" if self.df_studies.at[study_id, "completed"] else "#E0E0E0"
        return "#E0E0E0"
    
    def LoadTemplateActors(self, idx):
        surf_idx = self.current_actor_index - (5 - idx)
        self.main_mapper.SetInputData(self.template_surf[surf_idx])
    def NextTemplateActors(self):

        bottom = self.renderer_dict["bottom"]

        for ren_idx, ren_d in enumerate(bottom):
            ren_d["mapper"].SetInputData(self.template_surf[self.current_actor_index])
            
            self.current_actor_index = self.current_actor_index + 1 

            if self.current_actor_index == len(self.template_surf):
                self.current_actor_index = 0
        
    def PreviousTemplateActors(self):
            
        bottom = self.renderer_dict["bottom"]
        
        self.current_actor_index = self.current_actor_index - 2*len(bottom)

        if self.current_actor_index < 0:
            self.current_actor_index = len(self.template_surf) + self.current_actor_index

        self.NextTemplateActors()