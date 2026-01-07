"""Extract delta module."""

import pandas as pd
import argparse
from utils.functions import parse_deltas


def main():
    parser = argparse.ArgumentParser(description="The initial tool for extracting delta information out of DeltaLog.")
    
    parser.add_argument('deltas_path', type=str, help="Path to the Deltalog extracted csv file.")
    parser.add_argument('comments_path', type=str, help="Path to the r/changemyview comments jsonl dump.")
    parser.add_argument('output_path', type=str, help="CSV output file for the fixed version.")

    args = parser.parse_args()

    deltas_path = args.deltas_path
    comments_path = args.comments_path
    output_path = args.output_path

    deltas_df = pd.read_csv(deltas_path)

    # TODO: fix from here



if __name__ == "__main__":
    main()
