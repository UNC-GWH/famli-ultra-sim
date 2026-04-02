#!/usr/bin/env python

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def infer_default_split_paths(main_csv: str):
    """
    Given the main CSV path, infer default paths for train/valid/test:
    <base>_train_train.csv, <base>_train_test.csv, <base>_test.csv
    """
    p = Path(main_csv)
    base = p.with_suffix("")
    train = p.with_name(base.name + "_train_train.csv")
    valid = p.with_name(base.name + "_train_test.csv")
    test = p.with_name(base.name + "_test.csv")
    return train, valid, test,


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Update train/valid/test splits from a master CSV. "
            "Existing groups keep their split; new groups are split 80/20 "
            "train/test, and train is further split 90/10 into train/valid."
        )
    )

    # Required main CSV
    parser.add_argument(
        "--csv",
        help="Main CSV file containing all samples (old + new).",
        required=True,
    )

    # Optional explicit split paths
    parser.add_argument(
        "--csv_train",
        help="Train split CSV. Default: <csv>_train_train.csv",
        default=None,
    )
    parser.add_argument(
        "--csv_valid",
        help="Validation split CSV. Default: <csv>_train_test.csv",
        default=None,
    )
    parser.add_argument(
        "--csv_test",
        help="Test split CSV. Default: <csv>_test.csv",
        default=None,
    )

    # grouping column
    parser.add_argument(
        "--group_by",
        default="pid",
        help="Column name defining the group for splitting (e.g., pid, study_id). "
             "All rows of the same group stay in the same split. Default: pid",
    )

    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite existing split CSVs instead of writing *_updated.csv.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to split new groups. Default: 0",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Infer default split paths if not provided
    if args.csv_train is None or args.csv_valid is None or args.csv_test is None:
        default_train, default_valid, default_test = infer_default_split_paths(args.csv)

        csv_train = Path(args.csv_train) if args.csv_train else default_train
        csv_valid = Path(args.csv_valid) if args.csv_valid else default_valid
        csv_test = Path(args.csv_test) if args.csv_test else default_test
    else:
        csv_train = Path(args.csv_train)
        csv_valid = Path(args.csv_valid)
        csv_test = Path(args.csv_test)

    group_col = args.group_by

    # Load data
    df_main = pd.read_csv(args.csv)
    df_train_old = pd.read_csv(csv_train)
    df_valid_old = pd.read_csv(csv_valid)
    df_test_old = pd.read_csv(csv_test)

    if group_col not in df_main.columns:
        raise KeyError(f"Column '{group_col}' not found in main CSV: {args.csv}")

    for split_name, df in [
        ("train", df_train_old),
        ("valid", df_valid_old),
        ("test", df_test_old),
    ]:
        if group_col not in df.columns:
            raise KeyError(
                f"Column '{group_col}' not found in {split_name} CSV. "
                f"Check {split_name} file: "
                f"{ {'train': csv_train, 'valid': csv_valid, 'test': csv_test}[split_name] }"
            )

    # Group sets
    groups_all = set(df_main[group_col].unique())
    groups_train_old = set(df_train_old[group_col].unique())
    groups_valid_old = set(df_valid_old[group_col].unique())
    groups_test_old = set(df_test_old[group_col].unique())

    # Sanity check overlaps
    overlap_tv = groups_train_old & groups_valid_old
    overlap_tt = groups_train_old & groups_test_old
    overlap_vt = groups_valid_old & groups_test_old
    if overlap_tv or overlap_tt or overlap_vt:
        print("WARNING: Found overlapping groups between splits.")
        if overlap_tv:
            print(f"  Train & Valid overlap (sample): {list(overlap_tv)[:5]}")
        if overlap_tt:
            print(f"  Train & Test overlap (sample): {list(overlap_tt)[:5]}")
        if overlap_vt:
            print(f"  Valid & Test overlap (sample): {list(overlap_vt)[:5]}")

    known_groups = groups_train_old | groups_valid_old | groups_test_old
    new_groups = groups_all - known_groups

    print(f"Total groups in main CSV: {len(groups_all)}")
    print(f"Existing groups in splits: {len(known_groups)}")
    print(f"New groups found:          {len(new_groups)}")

    # If there are new groups, split them 80/20 train/test and 90/10 train/valid
    new_train_groups = set()
    new_valid_groups = set()
    new_test_groups = set()

    if new_groups:
        rng = np.random.default_rng(args.seed)
        new_groups_list = np.array(list(new_groups))
        rng.shuffle(new_groups_list)

        n_new = len(new_groups_list)
        n_new_test = int(round(0.2 * n_new))  # 20% → test
        test_idx_end = n_new_test

        new_test_groups = set(new_groups_list[:test_idx_end])
        remaining = new_groups_list[test_idx_end:]

        # remaining go to the "train side" (train + valid)
        n_remaining = len(remaining)
        # 10% of remaining → valid (since remaining is 80% of total new groups,
        # this corresponds to 0.8 * 0.1 = 8% overall valid among new groups)
        n_new_valid = int(round(0.1 * n_remaining))
        valid_idx_end = n_new_valid

        new_valid_groups = set(remaining[:valid_idx_end])
        new_train_groups = set(remaining[valid_idx_end:])

        print("New groups split summary:")
        print(f"  New train groups: {len(new_train_groups)}")
        print(f"  New valid groups: {len(new_valid_groups)}")
        print(f"  New test groups:  {len(new_test_groups)}")

    # Final group sets: keep old ones, add new ones
    final_train_groups = groups_train_old | new_train_groups
    final_valid_groups = groups_valid_old | new_valid_groups
    final_test_groups = groups_test_old | new_test_groups

    # Filter main df to build updated splits
    df_train_new = df_main[df_main[group_col].isin(final_train_groups)].copy()
    df_valid_new = df_main[df_main[group_col].isin(final_valid_groups)].copy()
    df_test_new = df_main[df_main[group_col].isin(final_test_groups)].copy()

    print("Updated split sizes (rows):")
    print(f"  Train: {len(df_train_new)}")
    print(f"  Valid: {len(df_valid_new)}")
    print(f"  Test:  {len(df_test_new)}")

    # Output paths
    if args.inplace:
        out_train = csv_train
        out_valid = csv_valid
        out_test = csv_test
    else:
        out_train = csv_train.with_name(csv_train.stem + "_updated.csv")
        out_valid = csv_valid.with_name(csv_valid.stem + "_updated.csv")
        out_test = csv_test.with_name(csv_test.stem + "_updated.csv")

    # Write CSVs
    df_train_new.to_csv(out_train, index=False)
    df_valid_new.to_csv(out_valid, index=False)
    df_test_new.to_csv(out_test, index=False)

    print("Wrote updated splits to:")
    print(f"  Train: {out_train}")
    print(f"  Valid: {out_valid}")
    print(f"  Test:  {out_test}")


if __name__ == "__main__":
    main()
