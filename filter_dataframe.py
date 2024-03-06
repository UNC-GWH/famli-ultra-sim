import pandas as pd
import argparse 


def filter_studies(df, tags):
    # Function to check if all tags are present
    def has_all_tags(group):
        group_tags = set(group['tag'])
        return set(tags).issubset(group_tags)

    # Group by 'study_id' and filter
    valid_studies = df.groupby('study_id').filter(has_all_tags)
    
    return valid_studies


def main(args):
    df = pd.read_csv(args.csv)

    df_filtered = filter_studies(df, args.tags)

    df_filtered.to_csv(args.out, index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Filter data frame', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--csv', type=str, help='CSV file with columns', required=True)
    parser.add_argument('--out', type=str, help='Filtered output', required=True)
    parser.add_argument('--tags', type=str, nargs='+', help='Set of tags', default=["L1", "L0", "M", "R0", "R1", "C1", "C2", "C3", "C4"])

    args = parser.parse_args()

    main(args)

# Example usage:
# tags = ["L1", "L0", "M", "R0", "R1", "C1", "C2", "C3", "C4"]
# filtered_df = filter_studies(df, tags)
