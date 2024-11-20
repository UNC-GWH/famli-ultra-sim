import pandas as pd
from pathlib import Path
import argparse
import os

def main(args):

    df = []	

    for f in Path(args.dir).rglob('*.nrrd'):    	

        f = str(f)
        img_fn = f.replace(os.path.normpath(args.dir) + '/', '')

        ignore_size_info = 0
        group = os.path.normpath(os.path.dirname(img_fn))
        if group in args.ignore_size_info:
            ignore_size_info = 1

        if not os.path.basename(os.path.dirname(img_fn)) in args.ignore:
            df.append({"img_fn": os.path.join(os.path.basename(os.path.normpath(args.dir)), img_fn), 'group': group, 'ignore_size_info': ignore_size_info})

    df = pd.DataFrame(df).sort_values(['group', 'img_fn']).reset_index(drop=True).reset_index()
    
    df["label"] = df["index"] + 1	
    df.to_csv(os.path.normpath(args.dir) + ".csv", index=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create a CSV file that has the index of structures', formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--dir', type=str, help='Directory with nrrd files', required=True)   
    parser.add_argument('--ignore_size_info', type=str, help='Add column ignore size info', nargs="+", default=["lady"])    
    parser.add_argument('--ignore', type=str, help='Add column ignore elements', nargs="+", default=["probe", "subcorticals"])    
    
    args = parser.parse_args()

    main(args)