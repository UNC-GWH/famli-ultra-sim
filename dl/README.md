# README

## Export mesh

CUDA_VISIBLE_DEVICES=1 blender -b /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/Pregnant_Fetus_Uterus_Blend_2-82/Pregnant_Fetus.blend --python /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/famli-ultra-sim/export_mesh.py -- --export_dir FAM-202-1960-2_mesh

## Create Stencil

python /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/famli-ultra-sim/create_stencil.py --dir ./FAM-202-1960-2_mesh

## Resample plane stencil
python /mnt/famli_netapp_shared/C1_ML_Analysis/src/US-famli/src/py/dl_torch/resample.py --img FAM-202-1960-2_mesh/probe/ultrasound_plane.nrrd --size 256 256 1 --fit_spacing 1 --out  FAM-202-1960-2_mesh/probe/ultrasound_plane_resampled.nrrd

## Create CSV and merge images

 python /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/famli-ultra-sim/merge_images.py --csv FAM-202-1960-2_mesh.csv --min_size 512 --pad 0.01 --a_min 0 --a_max 255

 ## Export probe params

 python /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/famli-ultra-sim/export_probe_params_wrapper.py --img FAM-202-1960-2_20211019_033119/BPD.nrrd --copy_rl FAM-202-1960-2
 --probe_fn FAM-202-1960-2_mesh/probe/ultrasound_plane_resampled.nrrd --out  FAM-202-1960-2_mesh_probe_params/BPD

 ## Image sampling with probe params

 /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/famli-ultra-sim/image_sampling_wrapper.sh FAM-202-1960-2_mesh_probe_params FAM-202-1960-2_mesh.nrrd
 python /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/famli-ultra-sim/image_sampling.py --probe_params $probe_params_dir/$ppd --img $img --out $out/$ppd