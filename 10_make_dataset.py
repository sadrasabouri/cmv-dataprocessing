"""Make dataset module."""

import pandas as pd
from tqdm import tqdm
import json
import pickle
import argparse
from utils.functions import get_indexed_comment_maps, get_chat_hist
from utils.functions import fix_deltas, extract_deltas, get_delta


def main():
    parser = argparse.ArgumentParser(description="The second tool for creating dataset by joining datasources.")
    
    parser.add_argument('submissions_path', type=str, help="Path to the r/changemyview submission jsonl dump")
    parser.add_argument('comments_path', type=str, help="Path to the r/changemyview comments jsonl dump")
    parser.add_argument('deltas_path', type=str, help="Path to the DeltaLog csv delta files")
    parser.add_argument('output_path', type=str, help="Dataset jsonl path")
    
    parser.add_argument('--use-cache', action="store_true", help="Flag that lets the program use the generated cache", default=None)

    args = parser.parse_args()

    submissions_path = args.submissions_path
    comments_path = args.comments_path
    deltas_path = args.deltas_path
    output_path = args.output_path

    if args.use_cache:
        with open('~pid_cid2comment.pkl', 'rb') as f:
            pid_cid2comment = pickle.load(f)
        with open('~pid2comment.pkl', 'rb') as f:
            pid2comment = pickle.load(f)
        with open('~deltas_dict.pkl', 'rb') as f:
            deltas = pickle.load(f)
    else:
        pid_cid2comment, pid2comment = get_indexed_comment_maps(comments_path)
        deltas_df = pd.read_csv(deltas_path)
        deltas_df, _, _ = fix_deltas(deltas_df, pid_cid2comment)
        deltas = extract_deltas(deltas_df)
        del deltas_df

    # Constructing the dataset
    dataset = {
        'post_id': [], 'post_title': [], 'post_author': [], 'post_url': [], 'post_ups': [], 'post_downs': [], 'post_score': [], 'post_created_utc': [],
        'comment_id': [], 'comment_author': [], 'comment_ups': [], 'comment_downs': [], 'comment_score': [], 'comment_author_flair_text': [], 'comment_created_utc': [],
        'conversation': [], 'conversation_authors': [], 'conversation_length': [], 'conversation_ids': [],
        'delta_count': [], 'is_op_delta': [],
    }
    with open(submissions_path, 'r') as f:
        for line in tqdm(f, desc="Processing submissions ..."):
            post = json.loads(line.strip())
            for comment in pid2comment.get(post.get('id'), []):
                history, history_authors, history_ids = get_chat_hist(post.get('id'), comment.get('parent_id'), pid_cid2comment)
                conversation = [post.get('selftext'), *history, comment.get('body')]
                conversation_authors = [post.get('author'), *history_authors, comment.get('author')]
                conversation_ids = [comment.get('link_id'), *history_ids, comment.get('id')]
                delta_info = get_delta(post.get('id'), comment.get('id'), deltas)
                
                dataset['post_id'].append(post.get('id'))
                dataset['post_title'].append(post.get('title'))
                dataset['post_author'].append(post.get('author'))
                dataset['post_url'].append(post.get('url'))
                dataset['post_ups'].append(post.get('ups'))
                dataset['post_downs'].append(post.get('downs'))
                dataset['post_score'].append(post.get('score'))
                dataset['post_created_utc'].append(post.get('created_utc'))

                dataset['comment_id'].append(comment.get('id'))
                dataset['comment_author'].append(comment.get('author'))
                dataset['comment_ups'].append(comment.get('ups'))
                dataset['comment_downs'].append(comment.get('downs'))
                dataset['comment_score'].append(comment.get('score'))
                dataset['comment_author_flair_text'].append(comment.get('author_flair_text'))
                dataset['comment_created_utc'].append(comment.get('created_utc'))

                dataset['conversation'].append(conversation)
                dataset['conversation_authors'].append(conversation_authors)
                dataset['conversation_length'].append(len(conversation))
                dataset['conversation_ids'].append(conversation_ids)

                dataset['delta_count'].append(delta_info['count'])
                dataset['is_op_delta'].append(delta_info['is_op_delta'])

    # Dump dataset
    print("Creating absolute dataset (cmv_delta.jsonl) ...")
    ALLOWED_KEYS = ["post_id", "post_title", "post_author", "post_url", "post_ups", "post_downs", "post_score", "post_created_utc",
                    "comment_id", "comment_author", "comment_ups", "comment_downs", "comment_score", "comment_author_flair_text", "comment_created_utc",
                    "conversation", "conversation_authors", "conversation_length", "conversation_ids",
                    "delta_count", "is_op_delta"]
    with open(output_path, 'w') as f:
        for i in range(len(dataset['post_id'])):
            record = {key: dataset[key][i] for key in dataset if key in ALLOWED_KEYS}
            f.write(json.dumps(record) + '\n')

    print("Creating relational dataset (cmv_delta_rel.jsonl) ...")
    ALLOWED_KEYS = ["post_id", "comment_id", "conversation_ids", "delta_count", "is_op_delta"]
    with open(output_path.replace('.jsonl', '_rel.jsonl'), 'w') as f:
        for i in range(len(dataset['post_id'])):
            record = {key: dataset[key][i] for key in dataset if key in ALLOWED_KEYS}
            f.write(json.dumps(record) + '\n')

if __name__ == '__main__':
    main()
