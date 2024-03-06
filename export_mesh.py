# exports each selected object into its own file

import bpy
import mathutils
import os
import sys

sys.path.append(os.path.dirname(__file__))
from argparse_blender import ArgumentParserForBlender


parser = ArgumentParserForBlender()
parser.add_argument("--export_dir", help="Export directory")

args = parser.parse_args()


for obj in bpy.data.objects:
    obj.select_set(False)

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

        export_mesh(obj, os.path.join(export_dir, collection_name), use_identity)

        obj.select_set(False)

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
iterate_collection("gestational", export_dir)
iterate_collection("probe", export_dir, use_identity=True)