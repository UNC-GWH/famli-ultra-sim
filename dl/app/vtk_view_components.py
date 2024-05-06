
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

from datetime import datetime

from multiprocessing import Pool, cpu_count

class ViewComponents:
    def __init__(self, mount_point=None, num_bottom_renderers=5, data_path="Groups/FAMLI/QCApps/3DQCApp/data"):
        
        self.study_image_data = {}
        self.mount_point = mount_point
        self.data_path = data_path

        self.study_ids = []
        self.current_study_idx = -1
        self.current_surf_idx = 0
        self.study_id = ""
        self.study_images = {}
        self.user = ""

        renderWindowInteractor = vtkRenderWindowInteractor()
        self.resliceViewer = vtkResliceImageViewer()
        self.resliceViewer.SetupInteractor(renderWindowInteractor)
        self.resliceViewer.GetRenderWindow().SetOffScreenRendering(1)
        img = vtk.vtkImageData()
        img.SetDimensions(1, 1, 1)
        img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        self.resliceViewer.SetInputData(img)

        self.renderer_dict = {}

        self.renderer_dict["main"] = createRenderer()
        
        self.main_actor, self.main_mapper = createActor(surf=None, return_mapper=True)


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

        self.template_surf = []
        self.renderer_dict["bottom"] = []
        self.current_actor_index = 0

        for i in range(num_bottom_renderers):
            ren_d = createRenderer()
            self.renderer_dict["bottom"].append(ren_d)

        if self.mount_point is not None:
            self.Initialize()
        
    def MakeCopyForUser(self):

        orig_df_fn = os.path.join(self.mount_point, self.data_path, "C_dataset_analysis_protocoltagsonly_gaboe230_ge_iq_train.csv")
        df_fn = os.path.join(self.mount_point, self.data_path, "C_dataset_analysis_protocoltagsonly_gaboe230_ge_iq_train_{user}.csv".format(user=self.user))

        if not os.path.exists(df_fn):
            df = pd.read_csv(orig_df_fn)
            df.to_csv(df_fn, index=False)

    def Initialize(self):

        self.df_fn = os.path.join(self.mount_point, self.data_path, "C_dataset_analysis_protocoltagsonly_gaboe230_ge_iq_train_{user}.csv".format(user=self.user))

        if not os.path.exists(self.df_fn):
            return False
        
        self.df = pd.read_csv(self.df_fn)

        ext = os.path.splitext(self.df_fn)[1]
        self.studies_fn = self.df_fn.replace(ext, "_studies" + ext)

        if not os.path.exists(self.studies_fn):
            self.df_studies = self.df[["study_id"]].drop_duplicates()
            self.df_studies.set_index("study_id", inplace=True)
            self.df_studies["tx"] = 0.0
            self.df_studies["ty"] = 0.0
            self.df_studies["tz"] = 0.0
            self.df_studies["w"] = 0.0
            self.df_studies["rx"] = 0.0
            self.df_studies["ry"] = 0.0
            self.df_studies["rz"] = 0.0
            self.df_studies["completed"] = 0
            self.df_studies["template_fn"] = ""
        else:
            self.df_studies = pd.read_csv(self.studies_fn, index_col="study_id")
        # print(self.df_studies)
        self.study_ids = self.df["study_id"].drop_duplicates().tolist()

        self.template_arr = [
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0499-5/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-1336-3/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-1398-3/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0447-5/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0615-3/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-1453-3/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0626-2/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0664-4/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0749-4/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-1485-1/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-1489-1/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0754-2/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-1491-2/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0941-1/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0795-1/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0869-1/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-1275-1/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-0950-4/fetus/Fetus_Model.stl",
            # "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/studies/FAM-025-1144-4/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_flexed_arms_cross_legs_uncross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_overflexed_arms_cross_legs_uncross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_neutral_arms_cross_legs_uncross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_neutral_arms_cross_legs_cross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_flexed_arms_cross_legs_cross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_overflexed_arms_cross_legs_cross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_neutral_arms_uncross_legs_cross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_flexed_arms_uncross_legs_cross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_overflexed_arms_uncross_legs_cross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_neutral_arms_uncross_legs_uncross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_flexed_arms_uncross_legs_uncross/fetus/Fetus_Model.stl",
            "Groups/FAMLI/Shared/C1_ML_Analysis/src/diffusion-models/blender/generic2/head_overflexed_arms_uncross_legs_uncross/fetus/Fetus_Model.stl"]

        for template_fn in self.template_arr:
            surf = readSurf(os.path.join(self.mount_point, template_fn))
            self.template_surf.append(surf)
        
        self.main_mapper.SetInputData(self.template_surf[0])
        self.renderer_dict["main"]["renderer"].AddActor(self.main_actor)
        self.renderer_dict["main"]["renderer"].ResetCamera()

        bottom = self.renderer_dict["bottom"]
        for ren_d in bottom:
            actor, mapper = createActor(surf=None, return_mapper=True)
            ren_d["mapper"] = mapper
            ren_d["renderer"].AddActor(actor)
        self.NextTemplateActors()

        for ren_d in bottom:            
            ren_d["renderer"].ResetCamera()

        self.grid_sweeps = {}
        model_grid_dir = os.path.join(self.mount_point, self.data_path, "scene_models/paths")
        for sweep_fn in glob.glob(os.path.join(model_grid_dir, '*.vtk')):
            key = os.path.basename(sweep_fn).split(".")[0].replace("_path", "")
            self.grid_sweeps[key] = readSurf(sweep_fn)

            actor = createActor(self.grid_sweeps[key])
            self.renderer_dict["main"]["renderer"].AddActor(actor)

        self.sweep_actors = []
        model_sweep_dir = os.path.join(self.mount_point, self.data_path, "data/scene_models/tube")
        for sweep_fn in glob.glob(os.path.join(model_sweep_dir, '*.stl')):
            self.sweep_actors.append(readCreateActor(sweep_fn))
        for actor in self.sweep_actors:
            actor.GetProperty().SetOpacity(0.5)
            self.renderer_dict["main"]["renderer"].AddActor(actor)

        ultrasound_fan_fn = os.path.join(self.mount_point, self.data_path, "scene_models/probe/ultrasound_fan_2d_iq.stl")

        self.ultrasound_fan_actor = readCreateActor(ultrasound_fan_fn)
        self.ultrasound_fan_actor_transform = vtkTransform()

        self.renderer_dict["main"]["renderer"].AddActor(self.ultrasound_fan_actor)

        return True
        

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
        return self.UpdateFan()

    def UpdateFan(self):
        slice_num = self.resliceViewer.GetSlice()

        if self.study_image_data and not self.tag == "" and self.tag in self.study_image_data:
            tau = slice_num/self.study_image_data[self.tag].GetDimensions()[-1]
            
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
        return slice_num

    def GetCurrentTransform(self, translation, rotation_angles):
        current_transform = None
        if hasattr(self, "df_studies") and self.study_id in self.df_studies.index:
            tx, ty, tz, w, rx, ry, rz = self.df_studies.loc[self.study_id, ["tx", "ty", "tz", "w", "rx", "ry", "rz"]].values
            current_transform = vtk.vtkTransform()
            current_transform.Translate(tx, ty, tz)
        
            # Apply rotations.
            current_transform.RotateWXYZ(w, rx, ry, rz)

        # Reset the transform to identity to apply new transformations
        main_transform = vtk.vtkTransform()
        main_transform.Identity()

        # Apply translation
        main_transform.Translate(*translation)
        
        # Apply rotations.
        main_transform.RotateX(rotation_angles[0])
        main_transform.RotateY(rotation_angles[1])
        main_transform.RotateZ(rotation_angles[2])

        if current_transform is not None:            
            main_transform.Concatenate(current_transform)

        return main_transform
    
    def OnTransformSliderChange(self, translation, rotation_angles):

        transform = self.GetCurrentTransform(translation, rotation_angles)
        self.main_actor.SetUserMatrix(transform.GetMatrix())
        

    def ApplyTransform(self, translation, rotation_angles):
        transform = self.GetCurrentTransform(translation, rotation_angles)

        tx, ty, tz = transform.GetPosition()
        w, rx, ry, rz = transform.GetOrientationWXYZ()

        self.df_studies.at[self.study_id, "tx"] = tx
        self.df_studies.at[self.study_id, "ty"] = ty
        self.df_studies.at[self.study_id, "tz"] = tz
        self.df_studies.at[self.study_id, "w"] = w
        self.df_studies.at[self.study_id, "rx"] = rx
        self.df_studies.at[self.study_id, "ry"] = ry
        self.df_studies.at[self.study_id, "rz"] = rz
        

    def LoadStudy(self, idx, study_id):
        self.current_study_idx = idx
        self.study_id = study_id
        self.study_image_data = {}
        study_images = {}

        for idx, row in self.df.query(f'study_id == "{study_id}"')[['tag', 'file_path']].iterrows():
            study_images[row['tag']] = {"text": row['tag'], "file_path": row['file_path']}
        

        # Organize the dictionary according to the acquisition order
        ordered_keys = ['M', 'R0', 'R1', 'L0', 'L1', "C1", "C2", "C3", "C4"] 
        self.study_images = {key: study_images[key] for key in ordered_keys if key in study_images}

        with Pool(cpu_count()) as p:
            img_data_d = p.map(readImageData, [os.path.join(self.mount_point, img["file_path"]) for img in self.study_images.values()])

        for img_d, key in zip(img_data_d, self.study_images.keys()):
            self.study_image_data[key] = createVtkImage(img_d)

        return [v for _, v in self.study_images.items()]
    
    def SetImage(self, tag):        
        if tag in self.study_image_data:            
            self.tag = tag
            self.resliceViewer.SetInputData(self.study_image_data[tag])
            self.resliceViewer.SetSlice(0)

    def LoadImg(self, tag):
        
        if tag in self.study_images:
            img_path = self.study_images[tag]["file_path"]

            img_path = os.path.join(self.mount_point, img_path)
            self.study_image_data[tag] = readImage(img_path)
    
    def UpdateStudyTemplate(self, study_id):
        if study_id in self.df_studies.index:
            template_fn = self.df_studies.at[study_id, "template_fn"]
            if template_fn in self.template_arr:
                surf_idx = self.template_arr.index(template_fn)
                print("Loading template:", template_fn, surf_idx)
                if surf_idx >= 0 and surf_idx < len(self.template_surf):
                    self.main_mapper.SetInputData(self.template_surf[surf_idx])
                else:
                    print("Template not found")

    def Save(self, translation, rotation_angles):
        # Check if the specified ID exists in the DataFrame
        if self.study_id in self.df_studies.index:

            transform = self.GetCurrentTransform(translation=translation, rotation_angles=rotation_angles)
            tx, ty, tz = transform.GetPosition()
            w, rx, ry, rz = transform.GetOrientationWXYZ()

            # Update the row columns as specified
            self.df_studies.at[self.study_id, "tx"] = tx
            self.df_studies.at[self.study_id, "ty"] = ty
            self.df_studies.at[self.study_id, "tz"] = tz
            self.df_studies.at[self.study_id, "w"] = w
            self.df_studies.at[self.study_id, "rx"] = rx
            self.df_studies.at[self.study_id, "ry"] = ry
            self.df_studies.at[self.study_id, "rz"] = rz
            self.df_studies.at[self.study_id, "completed"] = 1
            
            self.df_studies.at[self.study_id, "template_fn"] = self.template_arr[self.current_surf_idx]
            # Write the updated DataFrame back to the file
            self.df_studies.to_csv(self.studies_fn)
            self.df_studies.to_csv(self.studies_fn + str(datetime.today().strftime('%Y-%m-%d-%H')))
            
            print("Saved!")
    
    def GetStudyButtonColor(self, study_id):
        if study_id in self.df_studies.index:
            return "#00E676" if self.df_studies.at[study_id, "completed"] else "#E0E0E0"
        return "#E0E0E0"
    
    def FindFirstIncompleteStudy(self):        
        for idx, study_id in enumerate(self.study_ids):
            if study_id in self.df_studies.index and not self.df_studies.at[study_id, "completed"]:
                return idx, study_id            
        return 0, self.study_ids[0]
    
    def LoadTemplateActors(self, idx):
        self.current_surf_idx = self.current_actor_index - (5 - idx)
        self.main_mapper.SetInputData(self.template_surf[self.current_surf_idx])

    def NextTemplateActors(self):

        bottom = self.renderer_dict["bottom"]

        for ren_d in bottom:
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