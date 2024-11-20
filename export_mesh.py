# exports each selected object into its own file

# /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/blender-3.5.1-linux-x64/blender --background /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/Pregnant_Fetus_Uterus_Blend_2-82/Pregnant_Fetus_guibruss.blend --python /mnt/famli_netapp_shared/C1_ML_Analysis/src/diffusion-models/famli-ultra-sim/export_mesh.py -- --export_dir /mnt/famli_netapp_shared/C1_ML_Analysis/src/diffusion-models/blender

import bpy
import mathutils
import os
import sys
import vtk

sys.path.append(os.path.dirname(__file__))
from argparse_blender import ArgumentParserForBlender


parser = ArgumentParserForBlender()
parser.add_argument("--export_dir", help="Export directory")
parser.add_argument("--rig_collection", default='rig_fetus',help="Collection name where the armature is located")
parser.add_argument("--rig", default=None, help="Rig name")
parser.add_argument("--pose", default=None, help="Pose name")


args = parser.parse_args()


for obj in bpy.data.objects: 
    obj.select_set(False)


def apply_asset_pose_to_armature(armature_name, action_name):
    # Find the armature object by name
    armature = bpy.data.objects.get(armature_name)
    if armature is None:
        print(f"Armature '{armature_name}' not found.")
        return

    # Ensure the armature is selected and active
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)

    # Switch to Pose mode
    bpy.ops.object.mode_set(mode='POSE')

    # Find the action marked as an asset by name
    action = next((action for action in bpy.data.actions if action.asset_data and action.name == action_name), None)
    if action is None:
        print(f"Action '{action_name}' not found or not marked as an asset.")
        return

    # Apply the action as a pose to the armature
    # Assuming the action is an asset, we use the asset API to apply it
    if not hasattr(armature.animation_data, 'nla_tracks'):
        armature.animation_data_create()
    
    # Clear any existing poses
    bpy.ops.pose.transforms_clear()

    # Creating a new NLA track and strip to apply the pose
    track = armature.animation_data.nla_tracks.new()
    start_frame = int(action.frame_start)
    strip = track.strips.new(action.name, start_frame, action)
    strip.action = action

    print(f"Pose '{action_name}' from asset has been applied to armature '{armature_name}'.")




def export_mesh(obj, export_dir, use_identity=False):

    if obj.type == 'MESH':

        if use_identity:
            obj.parent = None
            obj.matrix_world = mathutils.Matrix.Identity(4)

        name = bpy.path.clean_name(obj.name)

        for mod in obj.modifiers:
            if mod.type == 'WIREFRAME':
                obj.modifiers.remove(mod)
        
        fn = os.path.join(export_dir, name).replace(" ", "_") + ".stl"
        
        if not os.path.exists(fn):
            out_dir = os.path.dirname(fn)
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        #    bpy.ops.export_scene.obj(filepath=fn + ".obj", use_selection=True)
            bpy.ops.export_mesh.stl(filepath=fn, use_selection=True)

            # Can be used for multiple formats
            # bpy.ops.export_scene.x3d(filepath=fn + ".x3d", use_selection=True)

            obj.select_set(False)

            print("written:", fn)


   
def iterate_collection(collection_name, export_dir, use_identity=False):
    # Get the collection from the Blender file
    collection = bpy.data.collections.get(collection_name)

    # Check if the collection exists
    if not collection:
        print(f"Collection '{collection_name}' not found.")
        return
    
    collection.hide_viewport = False
    
    # Iterate through the objects in the collection and print their names
    for obj in collection.objects:
        print(f"Object name: {obj.name}")
        # Deselect all objects
        # bpy.ops.object.select_all(action='DESELECT')
        # Select the object
        obj.hide_viewport = False
        obj.select_set(True)
        # Set the object as active
        bpy.context.view_layer.objects.active = obj

        if obj.type == 'CURVE':
            polyData = bezier_curve_to_vtk_polydata(obj)
            filename = os.path.join(os.path.join(export_dir, collection_name), obj.name + '.vtk')
            save_vtk_polydata(polyData, filename)
        else:
            export_mesh(obj, os.path.join(export_dir, collection_name), use_identity)

        obj.select_set(False)

def bezier_curve_to_vtk_polydata(curve_obj):
    # Initialize VTK objects for points and lines
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    
    # Process each spline in the curve
    for spline in curve_obj.data.splines:
        
        for i, point in enumerate(spline.points):
        # for i, point in enumerate(spline.bezier_points):
            p = point.co
            points.InsertNextPoint(p[0], p[1], p[2])
            if i > 0:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i - 1)
                line.GetPointIds().SetId(1, i)
                lines.InsertNextCell(line)

    # Create a polydata object
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)

    return polyData

def save_vtk_polydata(polyData, filename):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polyData)
    writer.Write()
    print(f"written: {filename}")
        

# Replace 'YourCollectionName' with the name of the collection you want to iterate through

# Get the armature object
# armature = bpy.data.objects.get("rig_fetus")
# Check if the armature exists and get the name
# if armature is not None:    
#     export_dir = armature.animation_data.action.name
# else:
#     export_dir = "export_mesh"

if args.export_dir:
    export_dir = args.export_dir

# if args.rig and args.pose:
#     apply_asset_pose_to_armature(args.rig, args.pose)

#     for obj in bpy.data.objects: 
#         obj.select_set(False)

print("Export dir:", export_dir)



# iterate_collection("lady", export_dir)

iterate_collection("skeleton", export_dir)
iterate_collection("ribs", export_dir)
iterate_collection("arms", export_dir)
iterate_collection("legs", export_dir)
iterate_collection("skull", export_dir)
iterate_collection("cardiovascular", export_dir)
iterate_collection("brain", export_dir)
# iterate_collection("subcorticals", export_dir)
iterate_collection("bronchus", export_dir)
iterate_collection("visceral", export_dir)
iterate_collection("fetus", export_dir)
# iterate_collection("uterus", export_dir)
# iterate_collection("gestational", export_dir)

# iterate_collection("probe", export_dir, use_identity=True)
# iterate_collection("paths", export_dir)
# iterate_collection("tube", export_dir)
