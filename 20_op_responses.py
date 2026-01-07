"""Extract OP responses."""

import pandas as pd
import sys
from tqdm import tqdm
import os
import json
import pickle

# TODO: fix from here

submissions_path = sys.argv[1]
deltas_path = sys.argv[2]

print("Loading deltas ...", flush=True)
deltas_df = pd.read_csv(deltas_path)


print("Loading id indexed to comments files (~pid2comment.pkl) ...", flush=True)
with open('~pid2comment.pkl', 'rb') as f:
    pid2comment = pickle.load(f)
print("Loading id indexed to comments files (~pid_cid2comment.pkl) ...", flush=True)
with open('~pid_cid2comment.pkl', 'rb') as f:
    pid_cid2comment = pickle.load(f)

pid2post = {}
with open(submissions_path, 'r') as f:
    for line in tqdm(f, desc="Processing submissions ..."):
        post = json.loads(line.strip())
        pid2post[post.get('id')] = post

print("Saving id indexed to submissions files (~pid2post.pkl) ...", flush=True)
with open('~pid2post.pkl', 'wb') as f:
    pickle.dump(pid2post, f)


def get_comment_by_id(post_id: str, comment_id: str):
    if (post_id, comment_id) not in pid_cid2comment:
        return None
    return pid_cid2comment[(post_id, comment_id)]


DELTA_DEFAULT = {"is_op_delta": False, "count": 0}
deltas = {}
for i, delta in tqdm(deltas_df.iterrows(), desc="Making delta index ..."):
    the_comment = get_comment_by_id(delta.post_id, delta.in_comment)
    if the_comment is None:
        continue
    comment_author = the_comment.get('author')
    # note that here we need the ~66k different samples here which point to the OP's comment
    if comment_author != delta.to:
        post_id = delta.post_id
        comment_id = delta.in_comment
        if post_id is None or comment_id is None:
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

op_delta_given_comments = []
op_delta_not_given_comments = []
for pid, comments in pid2comment.items():
    for comment in comments:
        is_from_op = comment.get('author') == pid2post.get(pid, {}).get('author')
        if is_from_op:
            is_delta = get_delta(pid, comment.get('id'))['is_op_delta']
            if is_delta:
                op_delta_given_comments.append(comment)
            else:
                op_delta_not_given_comments.append(comment)

with open(os.path.join('op_responses', 'cmv_op_pos.jsonl'), 'w') as f:
    for c in op_delta_given_comments:
        f.write(json.dumps(c) + '\n')

with open(os.path.join('op_responses', 'cmv_op_neg.jsonl'), 'w') as f:
    for c in op_delta_not_given_comments:
        f.write(json.dumps(c) + '\n')
