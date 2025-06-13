import bpy
import os
import mathutils
import numpy as np
import json 
import random 
import math
import sys 

import bpy

# Set Cycles as the render engine
bpy.context.scene.render.engine = 'CYCLES'

# Set the compute device type (options: 'CUDA', 'OPTIX', 'HIP', 'ONEAPI')
prefs = bpy.context.preferences
cycles_prefs = prefs.addons['cycles'].preferences
cycles_prefs.compute_device_type = 'CUDA'  # or 'OPTIX', 'HIP', etc.

# IMPORTANT: Trigger device enumeration
cycles_prefs.get_devices()

# Enable all devices (you can also filter here)s
for device in cycles_prefs.devices:
    device.use = True

# Set the scene to use GPU
bpy.context.scene.cycles.device = 'GPU'

print("Using devices:")
for device in prefs.addons['cycles'].preferences.devices:
    print(f" - {device.name}: {'Enabled' if device.use else 'Disabled'}")

def nerf(args, output_dir):
    # === CONFIGURATION ===
    mesh_name = args.camera_pos_mesh  # Change to your mesh object name
    output_folder = output_dir  # Change to your output path
    camera_name = args.camera_name
    image_format = args.format

    def save_intrinsic_camera_parameters(cam, file_path):
        # Intrinsics (assuming default Blender values)
        focal_length = cam.data.lens
        sensor_width = cam.data.sensor_width
        resolution_x = bpy.context.scene.render.resolution_x
        resolution_y = bpy.context.scene.render.resolution_y
        scale = bpy.context.scene.render.resolution_percentage / 100

        fx = (focal_length / sensor_width) * (resolution_x * scale)
        fy = fx  # Square pixels
        cx = resolution_x * scale / 2
        cy = resolution_y * scale / 2
        
        camera_params = {}
        camera_params['fx'] = fx
        camera_params['fy'] = fy
        camera_params['cx'] = cx
        camera_params['cy'] = cy
        camera_params['width'] = resolution_x * scale
        camera_params['height'] = resolution_y * scale

        with open(file_path, 'w') as f:
            json.dump(camera_params, f)

    def save_extrinsic_camera_parameters(cam, file_path):
        # Extrinsics
        c2w = []
        
        for row in cam.matrix_world:
            c2w.append(list(row))
        
        
        camera_params = {}
        camera_params['c2w'] = c2w  
        
        with open(file_path, 'w') as f:
            json.dump(camera_params, f) 

    mesh_obj = bpy.data.objects[mesh_name]

    # Create camera if it doesn't exist
    if camera_name in bpy.data.objects:
        cam_obj = bpy.data.objects[camera_name]
    else:
        cam_data = bpy.data.cameras.new(camera_name)
        cam_obj = bpy.data.objects.new(camera_name, cam_data)
        bpy.context.collection.objects.link(cam_obj)

    # Make camera the active camera
    bpy.context.scene.camera = cam_obj

    # Iterate over vertices
    cam_params_path = os.path.join(output_folder, f"camera_intrinsic.json")
    save_intrinsic_camera_parameters(cam_obj, cam_params_path)
    for i, vert in enumerate(mesh_obj.data.vertices):


        cam_params_path = os.path.join(output_folder, f"camera_{i:04d}.json")

        if not os.path.exists(cam_params_path):
            world_coord = mesh_obj.matrix_world @ vert.co

            # Move camera
            cam_obj.location = world_coord

            # Save render
            bpy.context.scene.render.filepath = os.path.join(output_folder, f"render_{i:04d}.{image_format.lower()}")
            bpy.ops.render.render(write_still=True)

            # Save camera parameters
            save_extrinsic_camera_parameters(cam_obj, cam_params_path)


def random_rotate_rig(args, output_dir):

    rig_name = args.rig  # Change to your rig's object name

    if os.path.exists(os.path.join(output_dir, 'rig_params.json')):

        with open(os.path.join(output_dir, 'rig_params.json'), 'r') as f:
            rig_params = json.load(f)
        r2w = rig_params['r2w']
        # Apply to rig object
        rig = bpy.data.objects[rig_name]
        rig.matrix_world = mathutils.Matrix(r2w)
        print(f"Loaded rig '{rig_name}' from {output_dir}")

    else:
        
        angle_rad = random.uniform(-math.pi/4, math.pi/4)

        # Random unit axis
        axis = mathutils.Vector((
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        )).normalized()

        rotation_matrix = mathutils.Matrix.Rotation(angle_rad, 4, axis)

        # Apply to rig object
        rig = bpy.data.objects[rig_name]
        rig.matrix_world = rotation_matrix @ rig.matrix_world

        print(f"Rotated rig '{rig_name}' by {math.degrees(angle_rad):.2f}° around axis {axis}")
        print(f"New matrix_world:\n{rig.matrix_world.to_3x3()}")

        rig_params = {}

        r2w = []
        for row in rig.matrix_world:
            r2w.append(list(row))

        rig_params['r2w'] = r2w

        with open(os.path.join(output_dir, 'rig_params.json'), 'w') as f:
            json.dump(rig_params, f)

sys.path.append(os.path.dirname(__file__))
from argparse_blender import ArgumentParserForBlender


parser = ArgumentParserForBlender()
parser.add_argument("--out", help="Output directory")
parser.add_argument("--rig", default='rig_fetus',help="Collection name where the armature is located")
parser.add_argument("--start_r", default=0, type=int, help="Starting number of rotations")
parser.add_argument("--end_r", default=360, type=int, help="End number of rotations")
parser.add_argument("--camera_pos_mesh", default="Icosphere", help="Mesh for camera positions")
parser.add_argument("--camera_name", default="Camera", help="Camera name")
parser.add_argument("--format", default="PNG", help="Output format")

args = parser.parse_args()

for idx in range(args.start_r, args.end_r):
    output_dir = os.path.join(args.out, f"rotation_{idx:04d}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    random_rotate_rig(args, output_dir)
    nerf(args, output_dir)
