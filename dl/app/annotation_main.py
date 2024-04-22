
import multiprocessing
from trame.app import get_server
from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.widgets import vtk as vtk_widgets, vuetify, html

# Required for interactor initialization
# local rendering, but doesn't hurt to include it
import vtkmodules.vtkRenderingOpenGL2  # noqa
import vtk_view_components as vvc

import pandas as pd

from utils import *

from pathlib import Path
import json
import argparse

vc = vvc.ViewComponents()

# Initialize the trame server with a single page layout
server = get_server(client_type = "vue2")
state, ctrl, ui = server.state, server.controller, server.ui

state.study_btn_colors = {}
state.load_study_btn = []
state.studies_initialized = 0

def get_path_string_until_directory(full_path, target_directory):
    """
    Returns a single path string from the root until it reaches the target directory (inclusive).

    Args:
    full_path (str): The full path from which to extract the path.
    target_directory (str): The directory to stop at (included in the result).

    Returns:
    str: A path string from the root up to the target directory.
    """
    # Normalize the path (resolve any .., ., etc.)
    full_path = os.path.normpath(full_path)
    
    # Initialize the path result and temporary path variable
    result_path = ""

    # Split the path into components
    for part in full_path.split(os.sep):
        if result_path:
            # Append the next part of the path
            result_path = os.path.join(result_path, part)
        else:
            # This is for the first part, which might be a drive letter
            result_path = part
        
        # If the current part is the target directory, stop building the path
        if part == target_directory:
            break

    return result_path

def OnMouseWheel():
    vc.OnMouseWheel()
    ctrl.view_update()

def SaveConfig():
    # vc.mount_point = "/Volumes/med/GWH"
    mount_point = os.path.normpath(state.mount_point).replace(os.sep, "/")
    user = state.user

    home_dir = Path.home()
    json.dump({"mount_point": mount_point, "user": user}, open(home_dir / ".qc_app_config.json", "w"))
    
    vc.mount_point = mount_point
    vc.user = user
    vc.MakeCopyForUser()

    LoadStudies()

def LoadStudies():

    if vc.Initialize():
        StudiesContent()
        LoadStudy(0, vc.study_ids[0])

def LoadStudy(idx, study_id):
    
    if vc.current_study_idx != -1:
        state[f"study_btn_color_{vc.current_study_idx}"] = vc.GetStudyButtonColor(vc.study_id) # Set the color of the study button state completed/not completed
    
    img_tags = vc.LoadStudy(idx, study_id)
    img_tag = img_tags[0]["text"]

    state[f"study_btn_color_{idx}"] = "#42A5F5" # Set the color of the active study button

    vc.LoadImg(img_tag)
    vc.UpdateFan()
    
    tx, ty, tz, rx, ry, rz = vc.GetStudyTransform(study_id)
    vc.UpdateStudyTemplate(study_id)

    print(f"tx: {tx}, ty: {ty}, tz: {tz}, rx: {rx}, ry: {ry}, rz: {rz}")
    
    state.update({
        "img_tags": img_tags,
        "img_tag": img_tag,
        "tx": tx,
        "ty": ty,
        "tz": tz,
        "rx": rx,
        "ry": ry,
        "rz": rz,
    })
    ctrl.view_update()

def Save():
    vc.Save(state.tx, state.ty, state.tz, state.rx, state.ry, state.rz)

def NextTemplateActors():
    vc.NextTemplateActors()
    ctrl.view_update()

def PreviousTemplateActors():
    vc.PreviousTemplateActors()
    ctrl.view_update()

def NextChangeMainActor():
    vc.NextChangeMainActor()
    ctrl.view_update()

def PreviousChangeMainActor():
    vc.PreviousChangeMainActor()
    ctrl.view_update()
    
def LoadTemplateActors(idx):
    vc.LoadTemplateActors(idx)
    ctrl.view_update()

@state.change("tx")
def OnTransformSliderChangeTx(tx, **kwargs):
    translation = np.array([tx, state.ty, state.tz])
    rotation_angles = np.array([state.rx, state.ry, state.rz])
    vc.OnTransformSliderChange(translation, rotation_angles)
    
    ctrl.view_update()

@state.change("ty")
def OnTransformSliderChangeTy(ty, **kwargs):
    translation = np.array([state.tx, ty, state.tz])
    rotation_angles = np.array([state.rx, state.ry, state.rz])
    vc.OnTransformSliderChange(translation, rotation_angles)
    
    ctrl.view_update()

@state.change("tz")
def OnTransformSliderChangeTz(tz, **kwargs):
    translation = np.array([state.tx, state.ty, tz])
    rotation_angles = np.array([state.rx, state.ry, state.rz])
    vc.OnTransformSliderChange(translation, rotation_angles)
    
    ctrl.view_update()

@state.change("rx")
def OnTransformSliderChangeTx(rx, **kwargs):
    translation = np.array([state.tx, state.ty, state.tz])
    rotation_angles = np.array([rx, state.ry, state.rz])
    vc.OnTransformSliderChange(translation, rotation_angles)
    
    ctrl.view_update()

@state.change("ry")
def OnTransformSliderChangeTx(ry, **kwargs):
    translation = np.array([state.tx, state.ty, state.tz])
    rotation_angles = np.array([state.rx, ry, state.rz])
    vc.OnTransformSliderChange(translation, rotation_angles)
    
    ctrl.view_update()

@state.change("rz")
def OnTransformSliderChangeRz(rz, **kwargs):
    translation = np.array([state.tx, state.ty, state.tz])
    rotation_angles = np.array([state.rx, state.ry, rz])
    vc.OnTransformSliderChange(translation, rotation_angles)
    
    ctrl.view_update()

@state.change("img_tag")
def OnImgTagChange(img_tag, **kwargs):
    vc.LoadImg(img_tag)
    ctrl.view_update()

def StudiesContent():
    with state:  # Force to update state before template
        state.setdefault("studies_initialized", 1)
    with ui.customizable_slot.clear():
        with vuetify.VContainer(
                        fluid=True,
                        classes="fill-height"
                ):
            with vuetify.VList():
                    for idx, study_id in enumerate(vc.study_ids):
                        with vuetify.VListItem():
                            vuetify.VBtn(block=True, children=f"{study_id}", click=lambda study_id=study_id, idx=idx: LoadStudy(idx, study_id), color=(f"study_btn_color_{idx}", vc.GetStudyButtonColor(study_id)))

def MainContent():
    with vuetify.VContainer(
            fluid=True,
            classes="fill-height",
        ):
        with vuetify.VRow(style="height: 75%"):
            with vuetify.VCol(cols=6):
                with vuetify.VToolbar():
                    vuetify.VSelect(items=("img_tags", []), v_model=("img_tag", ""))

                with vtk_widgets.VtkRemoteView(vc.resliceViewer.GetRenderWindow(), 
                                                ref="img_view",
                                                interactor_events=("events", ["EndMouseWheel"]), 
                                                EndMouseWheel=OnMouseWheel) as view:
                    ctrl.view_update.add(view.update)
            with vuetify.VCol(cols=6):
                with vuetify.VToolbar():
                    with vuetify.VCol(cols=2):
                        vuetify.VSlider(
                            v_model=("tx", 0), # (var_name, initial_value)
                            min=-0.1, max=0.1, step=0.001,
                            hide_details=True, dense=True,
                            label="Tx"
                        )
                    with vuetify.VCol(cols=2):
                        vuetify.VSlider(
                            v_model=("ty", 0), # (var_name, initial_value)
                            min=-0.1, max=0.1, step=0.001,
                            hide_details=True, dense=True,
                            label="Ty"
                        )
                    with vuetify.VCol(cols=2):
                        vuetify.VSlider(
                            v_model=("tz", 0), # (var_name, initial_value)
                            min=-0.1, max=0.1, step=0.001,
                            hide_details=True, dense=True,
                            label="Tz"
                        )
                    
                    with vuetify.VCol(cols=2):
                        vuetify.VSlider(
                            v_model=("rx", 0), # (var_name, initial_value)
                            min=-180, max=180, step=1,
                            hide_details=True, dense=True,
                            label="Rx"
                        )
                    with vuetify.VCol(cols=2):
                        vuetify.VSlider(
                            v_model=("ry", 0), # (var_name, initial_value)
                            min=-180, max=180, step=1,
                            hide_details=True, dense=True,
                            label="Ry"
                        )
                    with vuetify.VCol(cols=2):
                        vuetify.VSlider(
                            v_model=("rz", 0), # (var_name, initial_value)
                            min=-180, max=180, step=1,
                            hide_details=True, dense=True,
                            label="Rz"
                        )
                with vtk_widgets.VtkLocalView(vc.renderer_dict["main"]["renderWindow"]) as view:
                    ctrl.view_update.add(view.update)
        with vuetify.VRow(style="height: 15%"):
            with vuetify.VCol(cols=1):
                with vuetify.VBtn(icon=True, click=PreviousTemplateActors):
                    vuetify.VIcon("mdi-arrow-left")
            for ren_idx, ren_d in enumerate(vc.renderer_dict["bottom"]):
                with vuetify.VCol(cols=2):
                    with vuetify.VBtn(icon=True, click=lambda ren_idx=ren_idx: LoadTemplateActors(ren_idx)):
                        vuetify.VIcon("mdi-upload-box")
                    with vtk_widgets.VtkLocalView(ren_d["renderWindow"]) as view:
                            ctrl.view_update.add(view.update)
            with vuetify.VCol(cols=1):
                with vuetify.VBtn(icon=True, click=NextTemplateActors):
                    vuetify.VIcon("mdi-arrow-right")

with SinglePageWithDrawerLayout(server) as layout:

    layout.title.set_text("3D Model QC App")

    with layout.toolbar:
        
        with vuetify.VBtn(icon=True, click=SaveConfig):
            vuetify.VIcon("mdi-open-in-app")
        vuetify.VTextField(label="Mount Point", v_model=("mount_point", None))
        vuetify.VTextField(label="User", v_model=("user", None))
        #     vuetify.VInput(accept=".csv", ref="file_studies", type="file", multiple=False)
        # vuetify.VFileInput(accept=".csv", ref="file_studies", prepend_icon="mdi-open-in-app", multiple=False, hide_details=True, dense=True, v_model=("csv_studies", None))

        with vuetify.VBtn(icon=True, click=Save):
            vuetify.VIcon("mdi-content-save")

    with layout.drawer as drawer:
        drawer.clear()
        ui.customizable_slot(layout)
                            
    with layout.content as content:
        content.clear()
        MainContent()



# Starting the server
if __name__ == "__main__":
    multiprocessing.freeze_support()    

    if os.path.exists(Path.home() / ".qc_app_config.json"):
        config = json.load(open(Path.home() / ".qc_app_config.json"))
        vc.mount_point = config["mount_point"]
        state.mount_point = config["mount_point"]

        vc.user = config["user"]
        state.user = config["user"]

        if vc.Initialize():
            StudiesContent()
            LoadStudy(0, vc.study_ids[0])
    server.start(exec_mode="desktop")
    # server.start()
