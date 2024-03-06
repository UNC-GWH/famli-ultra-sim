import bpy
import os
import SimpleITK as sitk
import numpy as np 
import mathutils
import pickle
import sys 

sys.path.append(os.path.dirname(__file__))
from argparse_blender import ArgumentParserForBlender

parser = ArgumentParserForBlender()
parser.add_argument("--probe_fn", help="Probe fan image")
parser.add_argument("--copy_location", help="Use the copy location modifier to copy the location of an object", default=None)
parser.add_argument("--copy_rotation", help="Use the copy rotation modifier to copy the rotation of an object", default=None)
parser.add_argument("--out", help="Output directory", default='out')
parser.add_argument("--mesh_name", help="Mesh source name in blender", default="probe")
parser.add_argument("--cycles-device")
parser.add_argument("--samples", help="Number of samples for cycles", type=int, default=1)

args = parser.parse_args()


bpy.data.scenes["Scene"].cycles.samples = args.samples

img_probe_fn = args.probe_fn
out_dir = args.out

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

img_probe = None
if os.path.exists(img_probe_fn):
    img_probe = sitk.ReadImage(img_probe_fn)
    img_probe = sitk.Cast(img_probe, sitk.sitkFloat64)

# def frame_change_post_handler_img_probe(scene):
    
#     scene.frame_set(scene.frame_current)
#     probe = bpy.data.objects['probe']

#     # print(probe)

#     probe_origin = np.array(probe.matrix_world.to_translation())
#     probe_direction = np.array(probe.matrix_world.to_3x3())

    
#     ref_spacing = img_spacing

#     img_probe_physical_size = np.array(img_probe.GetSize())*np.array(img_probe.GetSpacing())
#     img_probe_real_size = (img_probe_physical_size/img_spacing).astype(int)
#     img_probe_real_size[-1] = 1
#     ref_size = img_probe_real_size
#     print("Ref size:", ref_size)   
    
#     delta_origin = mathutils.Vector((0, -ref_size[1]*ref_spacing[1]/2.0, 0, 1))  # Example point

#     # Multiply the point by the full transformation matrix
#     ref_origin = probe.matrix_world @ delta_origin
#     # The result is also a 4D vector, so you might want to convert it back to 3D
#     ref_origin = ref_origin.to_3d()

#     print(probe_origin, probe_direction, ref_origin)
    
#     ref = img_probe
#     ref.SetOrigin(ref_origin)
#     ref.SetDirection(probe_direction.flatten().tolist())

#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(ref)
#     # Resample the input volume to extract the slice
#     slice_image = resampler.Execute(img)

#     if img_probe is not None:

#         # Set up the filter
#         noise_filter = sitk.AdditiveGaussianNoiseImageFilter()
#         noise_filter.SetMean(0.0)
#         noise_filter.SetStandardDeviation(0.001)
#         slice_image = noise_filter.Execute(slice_image)

#         abs_filter = sitk.AbsImageFilter()        
#         slice_image = abs_filter.Execute(slice_image)

#         multiply_filter = sitk.MultiplyImageFilter()
#         slice_image = multiply_filter.Execute(slice_image, img_probe)

#     print(slice_image)

#     out_fn = os.path.join(out_dir, str(scene.frame_current) + ".nrrd")
#     print("Writing:", out_fn)
#     # Save the extracted slice (optional)
#     sitk.WriteImage(slice_image, out_fn)

def changeFollowPath(mesh_object_name, new_target_curve_name):
    # Get the mesh object
    mesh_object = bpy.data.objects[mesh_object_name]
    # Get the 'Follow Path' constraint
    follow_path_constraint = next((c for c in mesh_object.constraints if c.type == 'FOLLOW_PATH'), None)
    # Check if the 'Follow Path' constraint exists
    if follow_path_constraint is not None:
        # Get the new target curve object
        new_target_curve = bpy.data.objects[new_target_curve_name]
        # Set the new target
        follow_path_constraint.target = new_target_curve
    else:
        print("Follow Path constraint not found")

def copyRotation(mesh_object_name, new_target_object_name):
    # Get the mesh object
    mesh_object = bpy.data.objects[mesh_object_name]

     # Check if the mesh object exists
    if mesh_object is None:
        print(f"Object named {mesh_object_name} not found.")
        return

    # Get the 'Copy Rotation' constraint
    copy_rotation_constraint = next((c for c in mesh_object.constraints if c.type == 'COPY_ROTATION'), None)
    # Check if the 'Copy Rotation' constraint exists
    if copy_rotation_constraint is not None:
        # Get the new target object
        new_target_object = bpy.data.objects[new_target_object_name]
        # Set the new target
        copy_rotation_constraint.target = new_target_object
        print(f"Target of 'Copy Rotation' constraint set to {new_target_object_name}.")
    else:
        print("Copy Rotation constraint not found")

def copyLocation(mesh_object_name, new_target_object_name):
    # Get the mesh object
    mesh_object = bpy.data.objects.get(mesh_object_name)

    # Check if the mesh object exists
    if mesh_object is None:
        print(f"Object named {mesh_object_name} not found.")
        return

    # Get the 'Copy Location' constraint
    copy_location_constraint = next((c for c in mesh_object.constraints if c.type == 'COPY_LOCATION'), None)

    # Check if the 'Copy Location' constraint exists
    if copy_location_constraint is not None:
        # Get the new target object
        new_target_object = bpy.data.objects.get(new_target_object_name)

        # Check if the new target object exists
        if new_target_object is None:
            print(f"Target object named {new_target_object_name} not found.")
            return

        # Set the new target
        copy_location_constraint.target = new_target_object
        print(f"Target of 'Copy Location' constraint set to {new_target_object_name}.")
    else:
        print("Copy Location constraint not found")


def removeConstraintTargets(mesh_object_name):
    # Get the mesh object
    mesh_object = bpy.data.objects.get(mesh_object_name)

    # Check if the mesh object exists
    if mesh_object is None:
        print(f"Object named {mesh_object_name} not found.")
        return

    # List of constraint types to remove targets from
    constraint_types = ['FOLLOW_PATH', 'COPY_ROTATION', 'COPY_LOCATION']

    # Iterate over the constraints and remove targets
    for constraint in mesh_object.constraints:
        if constraint.type in constraint_types:
            constraint.target = None
            print(f"Removed target from {constraint.type} constraint.")

def frame_change_post_handler(scene):
    
    scene.frame_set(scene.frame_current)
    probe = bpy.data.objects['probe']

    # print(probe)

    probe_origin = np.array(probe.matrix_world.to_translation())
    probe_direction = np.array(probe.matrix_world.to_3x3())
    
    ref_size = img_probe.GetSize()
    ref_spacing = img_probe.GetSpacing()
    print("Ref size:", ref_size)   
    
    delta_origin = mathutils.Vector((0, -ref_size[1]*ref_spacing[1]/2.0, -ref_size[2]*ref_spacing[2]/2.0, 1))  # Example point

    # Multiply the point by the full transformation matrix
    ref_origin = probe.matrix_world @ delta_origin
    # The result is also a 4D vector, so you might want to convert it back to 3D
    ref_origin = np.array(ref_origin.to_3d())

    print(probe_origin, probe_direction, ref_origin)

    probe_params = {
        "probe_origin": probe_origin,
        "probe_direction": probe_direction,
        "ref_size": ref_size,
        "ref_origin": ref_origin,
        "ref_spacing": ref_spacing
    }

    out_probe_params_fn = os.path.join(out_dir, str(scene.frame_current) + "_probe_params.pickle")
    pickle.dump(probe_params, open(out_probe_params_fn, 'wb'))


removeConstraintTargets(args.mesh_name)

if args.copy_rotation:
    copyRotation(args.mesh_name, args.copy_rotation)
if args.copy_location:
    copyLocation(args.mesh_name, args.copy_location)

bpy.app.handlers.render_post.append(frame_change_post_handler)

    