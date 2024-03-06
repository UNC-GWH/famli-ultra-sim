#!/bin/zsh

dir=$(normpath $1)
out=${dir}_must

ls $dir | parallel -j8 python /mnt/famli_netapp_shared/C1_ML_Analysis/src/blender/famli-ultra-sim/run_must.py --dir $dir/{} --out $out/{}