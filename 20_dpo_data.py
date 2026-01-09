"""
DPO Dataset Extraction Module.

Here we assume the following for the dataset inclusion criteria:
+ the comment does not have any predecessor --> .conversation_length == 2
+ for the chosen one it randomly samples from the ones with op given delta
+ for the rejected one it randomly samples from the ones with op not given delta

Output format is a jsonl with fields like:
{
    "prompt": [.post_title + .post_text],
    "chosen": [a comment which got delta],
    "rejected": [a comment which didn't get delta],
}
"""

import json
import argparse
import pandas as pd
from tqdm import tqdm
from utils.functions import is_non

def select_chosen_reject(df: pd.DataFrame) -> pd.Series:
    """
    Select a pair of (chosen, rejected) for each post.

    :param df: dataframe split
    """
    true_rows = df[df["is_op_delta"] == True]
    false_rows = df[df["is_op_delta"] == False]

    if true_rows.empty or false_rows.empty:
        return None  # drop posts without both

    chosen = true_rows.sample(1).iloc[0]
    reject = false_rows.sample(1).iloc[0]
    
    assert chosen['post_title'] == reject['post_title']
    assert chosen['conversation'][0] == reject['conversation'][0]
    post_title = chosen['post_title']
    post_text = chosen['conversation'][0]
    prompt = f"{post_title}\n\n{post_text}"
    # It is a strict condition; not having them increased data by 25% but that was not a good sacrifice
    if is_non(post_title) or is_non(post_text):
        prompt = None
    
    # TODO: should be better for multi-hop
    chosen = '\n'.join(chosen['conversation'][1:])
    reject = '\n'.join(reject['conversation'][1:])

    return pd.Series({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": reject,
    })


def main():
    parser = argparse.ArgumentParser(description="A tool for creating DPO training pairs.")
    
    parser.add_argument('cmv_delta_path', type=str, help="Path to the cmv_delta dataset json")
    parser.add_argument('output_path', type=str, help="Directory to which the output is going to save")
    parser.add_argument('--seed', type=int, help="Random seed for the shuffling", default=42)

    args = parser.parse_args()

    cmv_delta_path = args.cmv_delta_path
    output_path = args.output_path
    seed = args.seed

    one_hop_comments = []
    # to decrease the memory usage:
    with open(cmv_delta_path, 'r') as f:
        for line in tqdm(f, desc="Loading delta ..."):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get('conversation_length', -1) == 2:
                one_hop_comments.append(data)
    one_hop_comments = pd.DataFrame(one_hop_comments)
    dataset = (
        one_hop_comments
        .groupby("post_id")
        .apply(select_chosen_reject, include_groups=False)
        .dropna()
    )
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_df = dataset.iloc[:n_train]
    val_df   = dataset.iloc[n_train:n_train + n_val]
    test_df  = dataset.iloc[n_train + n_val:]

    base_name = output_path.replace('.jsonl', '')
    train_df.to_json(f"{base_name}_train.jsonl", lines=True, orient='records')
    val_df.to_json(f"{base_name}_val.jsonl", lines=True, orient='records')
    test_df.to_json(f"{base_name}_test.jsonl", lines=True, orient='records')


if __name__ == "__main__":
    main()
