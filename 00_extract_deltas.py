"""Extract delta module."""

import pandas as pd
import json
import argparse
from utils.functions import parse_deltas


def main():
    parser = argparse.ArgumentParser(description="The initial tool for extracting delta information out of DeltaLog.")
    
    parser.add_argument('data_path', type=str, help="Path to the Deltalog subreddit jsonl dump.")
    parser.add_argument('output_path', type=str, help="CSV output file path.")

    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path

    df_list = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            df_list.extend(parse_deltas(data.get('selftext', None)))
    df = pd.DataFrame.from_records(df_list, columns=['post_id','from','to','in_comment','count','prefix'])
    df.to_csv(output_path, index=False)


def test(text: str) -> None:
    """
    Test the module.
    
    :param text: input text
    """
    print(parse_deltas(text))


if __name__ == "__main__":
    main()
