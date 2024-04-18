
import multiprocessing
from trame.app import get_server
from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.widgets import vtk as vtk_widgets, vuetify


# Required for interactor initialization
# local rendering, but doesn't hurt to include it
import vtkmodules.vtkRenderingOpenGL2  # noqa
import vtk_view_components as vvc

import pandas as pd

from utils import *

vc = vvc.ViewComponents()

# Initialize the trame server with a single page layout
server = get_server(client_type = "vue2")
state, ctrl = server.state, server.controller

state.study_btn_colors = {}

def OnMouseWheel():
    vc.OnMouseWheel()
    ctrl.view_update()

def LoadStudy(idx, study_id):
    
    if vc.current_study_idx != -1:
        state[f"study_btn_color_{vc.current_study_idx}"] = vc.GetStudyButtonColor(vc.study_id) # Set the color of the study button state completed/not completed
    
    img_tags = vc.LoadStudy(idx, study_id)
    img_tag = img_tags[0]["text"]

    state[f"study_btn_color_{idx}"] = "#42A5F5" # Set the color of the active study button

    vc.LoadImg(img_tag)
    vc.UpdateFan()

    tx, ty, tz, rx, ry, rz = vc.GetStudyTransform(study_id)

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

with SinglePageWithDrawerLayout(server) as layout:

    layout.title.set_text("3D Model QC App")

    with layout.toolbar:
        # toolbar components
        pass

    with layout.drawer as drawer:
        with vuetify.VContainer(
                fluid=True,
                classes="fill-height",
            ):
            with vuetify.VList():
                for idx, study_id in enumerate(vc.study_ids):
                    with vuetify.VListItem():
                        vuetify.VBtn(block=True, children=f"{study_id}", click=lambda study_id=study_id, idx=idx: LoadStudy(idx, study_id), color=(f"study_btn_color_{idx}", vc.GetStudyButtonColor(study_id)))
                            
    with layout.content:
        with vuetify.VContainer(
                fluid=True,
                classes="fill-height",
            ):
            with vuetify.VRow(style="height: 75%"):
                with vuetify.VCol(cols=6):
                    with vuetify.VToolbar():
                        vuetify.VSelect(items=("img_tags", []), v_model=("img_tag", ""))
                        with vuetify.VBtn(icon=True, click=Save):
                            vuetify.VIcon("mdi-content-save")

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

# Starting the server
if __name__ == "__main__":
    multiprocessing.freeze_support()
    server.start(exec_mode="desktop")
    # server.start()