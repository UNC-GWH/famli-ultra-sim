import bpy
import bmesh
import os
import mathutils
import numpy as np
import sys
from mathutils import Matrix

sys.path.append(os.path.dirname(__file__))
from argparse_blender import ArgumentParserForBlender


parser = ArgumentParserForBlender()
parser.add_argument("--export_dir", help="Export directory", default="./animation_export")
parser.add_argument("--start_frame", help="Start frame", type=int, default=None)
parser.add_argument("--end_frame", help="End frame", type=int, default=None)

args = parser.parse_args()

# Set the export base directory

EXPORT_DIR = args.export_dir
os.makedirs(EXPORT_DIR, exist_ok=True)
# Set the collections you want to export

COLLECTIONS_TO_EXPORT = [
    "skeleton", "ribs", "arms", "legs", "skull", "cardiovascular",
    "brain", "bronchus", "visceral", "fetus", "lady"
]

# COLLECTIONS_TO_EXPORT = ["arms", "legs"]

# COLLECTIONS_TO_EXPORT = {
#     "skeleton": ["skeleton", "ribs", "arms", "legs", "skull"],
#     "cardiovascular": ["cardiovascular"],
#     "visceral": ["visceral"]
# }
scene = bpy.context.scene
start_frame = args.start_frame if args.start_frame is not None else scene.frame_start
end_frame = args.end_frame if args.end_frame is not None else scene.frame_end

def export_mesh(obj, export_dir):
    if obj.type != 'MESH':
        print(f"Skipping {obj.name} as it is not a mesh object.")
        return

    filepath = os.path.join(export_dir, f"{obj.name}.stl")
    if os.path.exists(filepath):
        return

    os.makedirs(export_dir, exist_ok=True)

    # Export
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True, forward_axis='Y', up_axis='Z')
    bpy.ops.wm.stl_export(filepath=filepath, export_selected_objects=True)
    
    


def export_sweeps(target_objects, obj, modifier, frame, rotation, factor_values):
    # Export loop
    for target in target_objects:
        sweep_dir = os.path.join(EXPORT_DIR, f"frame_{frame:04d}", f"sweep_{target.name}")
        os.makedirs(sweep_dir, exist_ok=True)
        for idx, factor in enumerate(factor_values):
            # Set inputs
            modifier["Socket_4"] = target       
            modifier["Socket_2"] = factor      
            modifier["Socket_3"] = rotation  # Rotation in degrees  
            obj.update_tag(refresh={'DATA'})

            # Evaluate mesh
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = obj.evaluated_get(depsgraph)
            mesh_eval = obj_eval.to_mesh()

            # Copy mesh data so we can export it
            mesh_copy = mesh_eval.copy()
            obj_eval.to_mesh_clear()

            # Create a temp export object
            export_obj = bpy.data.objects.new("ExportTemp", mesh_copy)
            bpy.context.collection.objects.link(export_obj)
            export_obj.matrix_world = obj.matrix_world.copy()

            # Deselect all, select only export_obj
            for o in bpy.context.selected_objects:
                o.select_set(False)
            export_obj.select_set(True)
            bpy.context.view_layer.objects.active = export_obj

            # Export
            filepath = os.path.join(sweep_dir, f"{idx}.obj")
            bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True)
            print(f"Exported {filepath}")

            # Cleanup
            bpy.data.objects.remove(export_obj)
            bpy.data.meshes.remove(mesh_copy)

def export_us_simulation_fan(export_dir, grid_size=256):
    print(f"Exporting ultrasound simulation fan to {export_dir}...")

    obj = bpy.data.objects['ultrasound_fan_2d']
    
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    
    bvh = mathutils.bvhtree.BVHTree.FromBMesh(bm, epsilon=0.001)

    ## Cast rays
    grid_obj = bpy.data.objects['ultrasound_grid']

    ray_direction = mathutils.Vector((0, 0, 1)) 

    # Output: label map and depth map
    
    label_map = np.full((grid_size, grid_size), fill_value=0, dtype=np.int32)
    hit_verts = []
    hit_ij = []

    for idx, vert in enumerate(grid_obj.data.vertices):
        origin = grid_obj.matrix_world @ vert.co

        i = idx // (grid_size + 1)
        j = idx % (grid_size + 1)
        hit = bvh.ray_cast(origin, ray_direction)
        if hit[0] is not None:
            label_map[i, j] = 1
            hit_verts.append(list(vert.co.copy()))
            hit_ij.append((i, j))

    
    np.save(os.path.join(export_dir, "ultrasound_grid_np.npy"), label_map)
    export_mesh(grid_obj, export_dir)
    
    np.save(os.path.join(export_dir, "ultrasound_fan_hit_verts.npy"), hit_verts)
    np.save(os.path.join(export_dir, "ultrasound_fan_hit_verts_ij.npy"), hit_ij)


def export_probe_paths(export_dir, collection_name="probe_paths", probe_name="iq_plus", num_steps=200):
    # THIS WILL FAIL NOW BECAUSE THE GRID HAS THICKNESS - WE USED THIS WITH MITSUBA RENDERING
    
    depsgraph = bpy.context.evaluated_depsgraph_get()

    export_dir = os.path.join(export_dir, collection_name)

    os.makedirs(export_dir, exist_ok=True)
    probe_obj = bpy.data.objects[probe_name]

    probe_mod = probe_obj.modifiers.get("GeometryNodes")

    collection = bpy.data.collections.get(collection_name)

    for obj in collection.objects:

        probe_path_name = obj.name
        probe_rotation_name = obj.name + "_R"

        probe_path_obj = bpy.data.objects[probe_path_name]
        probe_rotation_obj = bpy.data.objects[probe_rotation_name]


        probe_mod['Socket_2'] = probe_path_obj
        probe_mod['Socket_12'] = probe_rotation_obj
        locations = []
        rotations = []

        for s in range(num_steps):
            factor = s / (num_steps - 1)

            probe_mod['Socket_7'] = factor
            
            probe_obj.data.update()
            bpy.context.view_layer.update()

            probe_eval = probe_obj.evaluated_get(depsgraph)
            probe_mesh = probe_eval.to_mesh()

            location_attr = probe_mesh.attributes.get("location")
            loc = np.array(location_attr.data[0].vector)
            locations.append(loc)

            rotation_attr = probe_mesh.attributes.get("rotation")
            rot = np.array(Matrix(rotation_attr.data[0].value).to_3x3())  # Get 4x4 matrix
            rotations.append(rot)
        
        np.save(os.path.join(export_dir, f"{obj.name}.npy"), locations)
        np.save(os.path.join(export_dir, f"{obj.name}_rotations.npy"), rotations)


# HERE STARTS THE EXPORT PROCESS

for frame in range(start_frame, end_frame + 1):
    print(f"Exporting frame {frame}...")
    scene.frame_set(frame)
    bpy.context.view_layer.update()

    frame_export_dir = os.path.join(EXPORT_DIR, f"frame_{frame:04d}")

    export_probe_paths(frame_export_dir)

    for collection_name in COLLECTIONS_TO_EXPORT:
        collection = bpy.data.collections.get(collection_name)        
        if not collection:
            print(f"Collection '{collection_name}' not found.")
            continue

        export_dir = os.path.join(frame_export_dir, collection_name)
        for obj in collection.objects:            
            export_mesh(obj, export_dir)
    
    # for k in COLLECTIONS_TO_EXPORT.keys():
        # merge_and_export_with_modifiers(
        #     collection_names=COLLECTIONS_TO_EXPORT[k],
        #     export_path=os.path.join(frame_export_dir, f"{k}.obj"),
        #     export_format='OBJ'
        # )

    
    # The object with the geometry node modifier
    obj_name = "ultrasound_grid"
    obj = bpy.data.objects[obj_name]
    
    export_mesh(obj, frame_export_dir)

    obj_name = "ultrasound_fan_2d"
    obj = bpy.data.objects[obj_name]
    
    export_mesh(obj, frame_export_dir)




    # modifier_name = "GeometryNodes"
    # modifier = obj.modifiers.get(modifier_name)
    # if not modifier or modifier.type != 'NODES':
    #     raise RuntimeError("Expected a Geometry Nodes modifier on 'GeoMesh'")

    # sweep_length = 200
    # factor_values = [x/(sweep_length - 1) for x in range(sweep_length)]
    
    # target_objects = [bpy.data.objects["M"], bpy.data.objects["L0"], bpy.data.objects["L1"], bpy.data.objects["R0"], bpy.data.objects["R1"]]  # sweep paths
    # target_rotation = [math.pi/2.0, math.pi/2.0, 0]  # rotation in degrees
    # # export_sweeps(target_objects, obj, modifier, frame, rotation=target_rotation, factor_values=factor_values)

    # target_objects = [bpy.data.objects["C1"], bpy.data.objects["C2"], bpy.data.objects["C3"], bpy.data.objects["C4"]]  # sweep paths
    # target_rotation = [-2.0*math.pi, math.pi/2.0, 0]  # rotation in degrees
    # # export_sweeps(target_objects, obj, modifier, frame, rotation=target_rotation, factor_values=factor_values)

