
import torch

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

import plotly.express as px
import plotly.graph_objects as go

import mitsuba as mi
mi.set_variant("cuda_rgb")

import drjit as dr
from torchvision import transforms as T

import SimpleITK as sitk
import argparse


def majority_vote(tensor):
    """
    tensor: shape [..., N] with integer values
    """  

    # One-hot encode the values
    one_hot = torch.nn.functional.one_hot(tensor, num_classes=tensor.max() + 1)  # shape [..., N, C]
    
    # Sum across N dimension
    counts = one_hot.sum(dim=-2)  # shape [..., C]

    # Take argmax along class dimension
    majority = counts.argmax(dim=-1)  # shape [...]
    return majority


def main(args):

    mount_dir = args.mount_dir
    surf_df = pd.read_csv(os.path.join(mount_dir, 'shapes_intensity_map.csv'))    

    surf_intensity_map_mean = [0]
    surf_intensiry_map_std = [0.02]

    frame = args.frame

    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'path'},
        "light": {"type": "constant"},
        "sensor": {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0.5, -0.5, 0.5], target=[0, 0, 0], up=[0, 0, 1]
            ),
        }
    }

    print(f"Loading {mount_dir} for frame {frame}...")    

    for i, row in surf_df.iterrows():
        surf_fn = row['surf']
        surf_id = str(i+1)
        scene_dict[surf_id] = {
            'id': surf_id,
            'type': 'obj',
            'filename': os.path.join(mount_dir, frame, surf_fn),
            "face_normals": True
        }
        surf_intensity_map_mean.append(row['mean'])
        surf_intensiry_map_std.append(row['stddev'])

    surf_intensity_map_mean = np.array(surf_intensity_map_mean)
    surf_intensiry_map_std = np.array(surf_intensiry_map_std)

    # scene_dict['ultrasound_fan_2d'] = {
    #     'type': 'obj',
    #     'filename': os.path.join(mount_dir, frame, 'ultrasound_fan_2d.obj'),
    # }

    # scene_dict['ultrasound_grid'] = {
    #     'type': 'obj',
    #     'filename': os.path.join(mount_dir, frame, 'ultrasound_grid.obj'),
    # }

    scene = mi.load_dict(scene_dict)

    shapes_np = scene.shapes_dr().numpy()
    shape_id_map = np.array([0]*(max(shapes_np)+1))

    for s_dr, s in zip(shapes_np, scene.shapes_dr()):
        shape_id_map[s_dr] = int(s.id())


    sweeps = ["M",
    "L0",
    "L1",
    "R0",
    "R1",
    "C1",
    "C2",
    "C3",
    "C4"]

    grid_size = 256

    sweeps_np = []
    depth_maps_np = []

    transform_us = T.Compose([T.Lambda(lambda x: T.functional.rotate(x, angle=270)), T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])

    ultrasound_fan_ij = np.load(os.path.join(mount_dir, "ultrasound_fan_hit_verts_ij.npy"))
    ultrasound_fan_hit_verts = np.load(os.path.join(mount_dir, "ultrasound_fan_hit_verts.npy"))

    ultrasound_fan_hit_verts_mi = mi.Point3f(ultrasound_fan_hit_verts[:, 0],
                                            ultrasound_fan_hit_verts[:, 1],
                                            ultrasound_fan_hit_verts[:, 2])

    sweeps_np = []

    for sweep in sweeps:    
            
        probe_origins = np.load(os.path.join(mount_dir, frame, "probe_paths", sweep + ".npy"))
        sweep_np = []
        for o in probe_origins: 

            label_map = []        
                
            o_mi = mi.Point3f(o)

            ultrasound_fan_rotation_mi = mi.Transform4f()
            if sweep in ["M", "L0", "L1", "R0", "R1"]:
                ultrasound_fan_rotation_mi = mi.Transform4f().translate(o_mi).rotate(axis=[0, 1, 0], angle=90).rotate(axis=[1, 0, 0], angle=90)
            elif sweep in ["C1", "C2", "C3", "C4"]:
                ultrasound_fan_rotation_mi = mi.Transform4f().translate(o_mi).rotate(axis=[0, 1, 0], angle=90).rotate(axis=[1, 0, 0], angle=180)

            ultrasound_fan_hit_verts_transformed_mi = ultrasound_fan_rotation_mi @ ultrasound_fan_hit_verts_mi

            for idx in range(5):

                # Random directions on the unit sphere

                if idx == 0:
                    directions = o_mi - ultrasound_fan_hit_verts_transformed_mi 
                if idx == 1:
                    directions = ultrasound_fan_rotation_mi @ mi.Vector3f(0, 0, -1)
                else:
                    phi = np.random.uniform(0, 2*np.pi, size=len(ultrasound_fan_hit_verts))
                    costheta = np.random.uniform(-1, 1, size=len(ultrasound_fan_hit_verts))
                    sintheta = np.sqrt(1 - costheta**2)

                    dx = sintheta * np.cos(phi)
                    dy = sintheta * np.sin(phi)
                    dz = costheta

                    directions = mi.Vector3f(dx, dy, dz)

                rays = mi.Ray3f(ultrasound_fan_hit_verts_transformed_mi, directions)
                si = scene.ray_intersect(rays)

                hit_shapes = shape_id_map[si.shape.numpy()]

                # Create a label map for the ultrasound fan hit vertices
                lm = np.zeros((grid_size, grid_size), dtype=np.int32)
                lm[ultrasound_fan_ij[:, 0], ultrasound_fan_ij[:, 1]] = hit_shapes
                label_map.append(lm)

            label_map = np.stack(label_map, axis=-1)
            label_map = majority_vote(torch.tensor(label_map).cuda().to(torch.long)).unsqueeze(0)
            sweep_np.append(transform_us(label_map).squeeze(0).cpu().numpy())
        sweeps_np.append(np.stack(sweep_np, axis=0))
    sweeps_np = np.stack(sweeps_np)

    out_dir = os.path.join(args.out, frame)
    os.makedirs(out_dir, exist_ok=True)
    for s, sp in zip(sweeps, sweeps_np):
        out_fn = os.path.join(out_dir, f"{s}_label.nrrd")
        print(f"Writing {out_fn}")
        
        img = sitk.GetImageFromArray(sp.astype(np.uint16))
        sitk.WriteImage(img, out_fn)
        
        us = surf_intensity_map_mean[sp] + np.random.normal(size=sp.shape)*surf_intensiry_map_std[sweeps_np[5]]*100
        us = np.clip(us, 0, 255).astype(np.uint8)
        out_fn_us = os.path.join(out_dir, f"{s}_us.nrrd")

        print(f"Writing {out_fn_us} done.")
        img_us = sitk.GetImageFromArray(us)
        sitk.WriteImage(img_us, out_fn_us)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount_dir", type=str, required=True)
    parser.add_argument("--frame", type=str, default="frame_0001")
    parser.add_argument("--out", type=str, required=True, help="Output directory for the sweeps")
    
    args = parser.parse_args()

    main(args)

