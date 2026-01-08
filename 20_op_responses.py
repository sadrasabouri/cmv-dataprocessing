"""Extract OP responses."""

import pandas as pd
import sys
from tqdm import tqdm
import os
import json
import pickle
import argparse
from utils.functions import get_comment_by_id
from utils.functions import extract_indexed_comment_maps, extract_indexed_post_map
from utils.functions import get_delta, fix_deltas, extract_deltas


def main():
    parser = argparse.ArgumentParser(description="The second tool for creating dataset by joining datasources.")
    
    parser.add_argument('submissions_path', type=str, help="Path to the r/changemyview submission jsonl dump")
    parser.add_argument('comments_path', type=str, help="Path to the r/changemyview comments jsonl dump")
    parser.add_argument('deltas_path', type=str, help="Path to the DeltaLog csv delta files (it should be initial files and not fixed)")
    
    parser.add_argument('--use-cache', action="store_true", help="Flag that lets the program use the generated cache", default=False)

    args = parser.parse_args()

    submissions_path = args.submissions_path
    comments_path = args.comments_path
    deltas_path = args.deltas_path

    deltas_df = pd.read_csv(deltas_path)
    if args.use_cache:
        with open('~pid_cid2comment.pkl', 'rb') as f:
            pid_cid2comment = pickle.load(f)
        with open('~pid2comment.pkl', 'rb') as f:
            pid2comment = pickle.load(f)
    else:
        pid_cid2comment, pid2comment = extract_indexed_comment_maps(comments_path)
    pid2post = extract_indexed_post_map(submissions_path)

    _, deltas_df, _ = fix_deltas(deltas_df, pid_cid2comment, save=False)
    deltas = extract_deltas(deltas_df, save=False)
    
    op_delta_given_comments = []
    op_delta_not_given_comments = []
    for pid, comments in pid2comment.items():
        for comment in comments:
            is_from_op = comment.get('author') == pid2post.get(pid, {}).get('author')
            if is_from_op:
                is_delta = get_delta(pid, comment.get('id'), deltas)['is_op_delta']
                if is_delta:
                    op_delta_given_comments.append(comment)
                else:
                    op_delta_not_given_comments.append(comment)

    if not os.path.exists('op_responses'):
        os.makedirs('op_responses')

    with open(os.path.join('op_responses', 'cmv_op_pos.jsonl'), 'w') as f:
        for c in op_delta_given_comments:
            f.write(json.dumps(c) + '\n')

    with open(os.path.join('op_responses', 'cmv_op_neg.jsonl'), 'w') as f:
        for c in op_delta_not_given_comments:
            f.write(json.dumps(c) + '\n')


if __name__ == "__main__":
    main()
