#!/bin/zsh


probe_params_dir=$()$1
img=$(normpath $2)
out=$(normpath ${img%.*})_sampling

ls $probe_params_dir | while read ppd; do 

    command="python /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/famli-ultra-sim/image_sampling.py --probe_params $probe_params_dir/$ppd --img $img --out $out/$ppd"
    echo $command
    eval $command
done;