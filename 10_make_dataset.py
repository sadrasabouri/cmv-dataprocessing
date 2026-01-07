"""Make dataset module."""

import pandas as pd
import sys
from tqdm import tqdm
import os
import json
import pickle
import argparse
from utils.functions import get_indexed_comment_maps

def main():
    parser = argparse.ArgumentParser(description="The second tool for creating dataset by joining datasources.")
    
    parser.add_argument('submissions_path', type=str, help="Path to the r/changemyview submission jsonl dump.")
    parser.add_argument('comments_path', type=str, help="Path to the r/changemyview comments jsonl dump.")
    parser.add_argument('deltas_path', type=str, help="Path to the DeltaLog csv delta files.")
    parser.add_argument('output_path', type=str, help="Dataset jsonl path.")
    
    parser.add_argument('--cache_pid_cid2comment', type=str, help="Path to the cached file for pid_cid2comment.", default=None)
    parser.add_argument('--cache_pid2comment', type=str, help="Path to the cached file for pid_cid2comment.", default=None)

    args = parser.parse_args()

    submissions_path = args.submissions_path
    comments_path = args.comments_path
    deltas_path = args.deltas_path
    output_path = args.output_path

    deltas_df = pd.read_csv(deltas_path)
    if args.cache_pid_cid2comment:
        with open(args.cache_pid_cid2comment, 'rb') as f:
            pid_cid2comment = pickle.load(f)
    if args.cache_pid2comment:
        with open(args.cache_pid2comment, 'rb') as f:
            pid2comment = pickle.load(f)
    if args.cache_pid_cid2comment is None or args.cache_pid2comment is None:
        pid_cid2comment, pid2comment = get_indexed_comment_maps(comments_path)

    def get_comment_by_id(post_id: str, comment_id: str):
        if (post_id, comment_id) not in pid_cid2comment:
            return None
        return pid_cid2comment[(post_id, comment_id)]

    # TODO: fix from here
    
    # fix deltas possible issues (~66k of deltas out of 94k)
    # EXPLANATION: in some cases delta-log-bot reported delta to the comment which gives the delta not the one who gets it.
    #   the proxy I found for such instances are to find cases where the author of the comment is not the same as the author to which the delta is given (reported)
    if not os.path.exists(deltas_path+'.fixed'):
        tobe_removed = []
        tobe_added = []
        for i, delta in tqdm(deltas_df.iterrows(), desc="Fixing deltalog-bot issue ..."):
            the_comment = get_comment_by_id(delta.post_id, delta.in_comment)
            if the_comment is None:
                continue
            comment_author = the_comment.get('author')
            if comment_author != delta.to:
                # print("[ISSUE]:", delta.post_id, delta.in_comment, f"{comment_author} (ACTUAL) != {delta.to} (Reported)")
                # Real delta receiver: delta.to
                tobe_removed.append(i)
                while True:
                    comment_id = the_comment.get('parent_id', '').split('_')[-1]
                    the_comment = get_comment_by_id(delta.post_id, comment_id)
                    if the_comment is None:
                        comment_id = None
                        break
                    the_author = the_comment.get('author')
                    if the_author == delta.to:
                        break
                tobe_added.append([delta.post_id, delta['from'], delta.to, comment_id, delta['count'], None]) # prefix is only for debug
        deltas_df = deltas_df.drop(tobe_removed)
        deltas_df = pd.concat([deltas_df,
                            pd.DataFrame.from_records(tobe_added, columns=['post_id','from','to','in_comment','count','prefix'])
                            ],ignore_index=True).reset_index()
        print(f"{len(tobe_removed)} rows fixed for deltabot bug!")
        deltas_df.to_csv(deltas_path+'.fixed', index=False)
    else:
        print("Fixed file exists. Retrieving it ...")
        deltas_df = pd.read_csv(deltas_path+'.fixed')


    DELTA_DEFAULT = {"is_op_delta": False, "count": 0}
    deltas = {}
    for i, delta in tqdm(deltas_df.iterrows(), desc="Making delta index files ..."):
        post_id = delta.post_id
        comment_id = delta.in_comment
        if post_id is None or comment_id is None: # cases where comment is removed (~4k of deltas out of 94k)
            continue
        
        if (post_id, comment_id) not in deltas:
            deltas[(post_id, comment_id)] = DELTA_DEFAULT.copy()
        deltas[(post_id, comment_id)]["count"] += int(delta['count'])
        if delta['from'] == "OP":
            deltas[(post_id, comment_id)]["is_op_delta"] = True

    def get_delta(post_id: str, comment_id: str):
        if (post_id, comment_id) not in deltas:
            return DELTA_DEFAULT
        return deltas[(post_id, comment_id)]

    print("Saving id indexed to comments files (~deltas_dict.pkl) ...", flush=True)
    with open('~deltas_dict.pkl', 'wb') as f:
        pickle.dump(deltas, f)

    del deltas_df


    # Constructing the dataset
    def get_chat_hist(post_id: str, parent_id: str):
        text, authors, ids = [], [], []
        while not parent_id == f"t3_{post_id}":
            comment = get_comment_by_id(post_id, parent_id.split('_')[-1])
            if comment is None: # parent comment is removed
                text.insert(0, None)
                authors.insert(0, None)
                ids.insert(0, None)
                break
            text.insert(0, comment.get('body'))
            authors.insert(0, comment.get('author'))
            ids.insert(0, comment.get('id'))

            parent_id = comment.get('parent_id')
        return text, authors, ids

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
                history, history_authors, history_ids = get_chat_hist(post.get('id'), comment.get('parent_id'))
                conversation = [post.get('selftext'), *history, comment.get('body')]
                conversation_authors = [post.get('author'), *history_authors, comment.get('author')]
                conversation_ids = [comment.get('link_id'), *history_ids, comment.get('id')]
                delta_info = get_delta(post.get('id'), comment.get('id'))
                
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
