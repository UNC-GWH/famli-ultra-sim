#!/usr/bin/env python
import pandas as pd
import argparse

def main(args):

    # Read the input parquet file
    df = pd.read_parquet(args.csv)
    
    # Create an empty list to store the sampled dataframes for each class
    balanced_dfs = []
    
    # Get the unique classes in the specified column
    unique_classes = df[args.class_column].unique()
    
    for cls in unique_classes:
        # Select the rows corresponding to the current class
        df_cls = df[df[args.class_column] == cls]
        # If the number of rows is less than n_samples, sample with replacement
        if len(df_cls) < args.n_samples:
            sampled_df = df_cls.sample(n=args.n_samples, replace=True, random_state=args.random_state)
        else:
            sampled_df = df_cls.sample(n=args.n_samples, random_state=args.random_state)
        balanced_dfs.append(sampled_df)
    
    # Concatenate the sampled groups
    balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
    
    # Write the balanced dataframe to a new parquet file
    balanced_df.to_parquet(args.out, index=False)
    print("New length", len(balanced_df))
    print(f"Balanced parquet file saved to {args.out}")
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Balance a parquet file by sampling n_samples rows per class.")
    parser.add_argument("--csv", help="Path to the input parquet file.", required=True)
    parser.add_argument("--out", help="Path to the output balanced parquet file.", default="out.parquet")
    parser.add_argument("--class_column", default="pred_class", help="Name of the class column (default: class_pred).")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples per class (default: 100).")
    parser.add_argument("--random_state", type=int, default=64, help="Random state for sampling (default: 64).")
    
    args = parser.parse_args()

    main(args)