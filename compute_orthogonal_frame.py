import numpy as np

import torch
import torch.nn.functional as F
import sys
import argparse
import pandas as pd

from pytorch3d.ops import (knn_points, 
                           knn_gather)

def compute_orthogonal_frame(points: torch.Tensor) -> torch.Tensor:    
    """
    Given a tensor of shape [B, 3, 3] representing three 3D points per batch,
    returns a tensor of shape [B, 3, 3] representing an orthogonal frame [x, y, z] for each batch.
    """
    p0 = points[:, 0]
    p1 = points[:, 1]
    p2 = points[:, 2]
    
    v1 = p1 - p0
    v2 = p2 - p0

    # Normalize x (first direction)
    x = F.normalize(v1, dim=1)

    # Compute z = normalized cross(v1, v2)
    z = F.normalize(torch.cross(v1, v2, dim=1), dim=1)

    # Compute y = cross(z, x)
    y = torch.cross(z, x, dim=1)

    # Stack the vectors as rows of the rotation matrix
    frame = torch.stack([x, y, z], dim=1)  # [B, 3, 3]

    return frame


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pc = np.load(args.pc)
    head_idx = np.load(args.head_idx)
    belly_idx = np.load(args.belly_idx)
    side_idx = np.load(args.side_idx)

    pc = torch.tensor(pc).to(device)
    
    num_surf = pc.shape[0]
    
    head_idx = torch.tensor(head_idx).to(device).expand(num_surf, -1, -1)
    belly_idx = torch.tensor(belly_idx).to(device).expand(num_surf, -1, -1)
    side_idx = torch.tensor(side_idx).to(device).expand(num_surf, -1, -1)

    head_k = knn_gather(pc, head_idx).squeeze(2)
    belly_k = knn_gather(pc, belly_idx).squeeze(2)
    side_k = knn_gather(pc, side_idx).squeeze(2)

    frame_points = torch.stack([torch.mean(belly_k, dim=1), 
                            torch.mean(head_k, dim=1),
                            torch.mean(side_k, dim=1)], dim=1)

    frame = compute_orthogonal_frame(frame_points).cpu().numpy()

    np.save(args.out, frame)

    if args.csv is not None:
        df = pd.read_csv(args.csv)
        df['frame'] = frame.reshape(num_surf, -1).tolist()

        csv_out = args.csv.replace('.csv', '_frame.csv')
        df.to_csv(csv_out, index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute orthogonal frame from 3D points")

    
    parser.add_argument('--pc', type=str, help='Path to all the point clouds', required=True)
    parser.add_argument('--head_idx', type=str, help='Path to head points', default="/mnt/raid/C1_ML_Analysis/simulated_data_export/fetus_rest_selected/head_idx.npy")
    parser.add_argument('--belly_idx', type=str, help='Path to belly points', default="/mnt/raid/C1_ML_Analysis/simulated_data_export/fetus_rest_selected/belly_idx.npy")
    parser.add_argument('--side_idx', type=str, help='Path to side points', default="/mnt/raid/C1_ML_Analysis/simulated_data_export/fetus_rest_selected/side_idx.npy")

    parser.add_argument('--csv', type=str, help='Path to the csv to append the frame', default=None)

    parser.add_argument('--out', type=str, help='Path to output file', default="out.npy")

    args = parser.parse_args()
    
    main(args)