import pandas as pd
import sys
from tqdm import tqdm
import os

submissions_path = sys.argv[1]
comments_path = sys.argv[2]
deltas_path = sys.argv[3]
output_path = sys.argv[4]

print("Loading deltas ...", flush=True)
deltas_df = pd.read_csv(deltas_path)
print("Loading submissions ...", flush=True)
submissions_df = pd.read_json(submissions_path, lines=True)
print("Loading comments ...", flush=True)
comments_df = pd.read_json(comments_path, lines=True)


def get_comment_by_id(post_id: str, comment_id: str):
    the_comment = comments_df[
        (comments_df['link_id'] == f"t3_{post_id}") &
        (comments_df['id'] == comment_id)
    ]
    if len(the_comment) == 0:
        return None
    if len(the_comment) != 1:
        print(f"[Error]: Multiple instances of post_id:{delta.post_id}, comment_id:{delta.in_comment}!")
    return the_comment.iloc[0]


# fix deltas possible issues
# EXPLANATION: in some cases delta-log-bot reported delta to the comment which gives the delta not the one who gets it.
#   the proxy I found for such instances are to find cases where the author of the comment is not the same as the author to which the delta is given (reported)
if not os.path.exists(deltas_path+'.fixed'):
    tobe_removed = []
    tobe_added = []
    for i, delta in tqdm(deltas_df.iterrows(), desc="Fixing deltalog-bot issue ..."):
        the_comment = get_comment_by_id(delta.post_id, delta.in_comment)
        if the_comment is None:
            continue
        comment_author = the_comment.author
        if comment_author != delta.to:
            # print("[ISSUE]:", delta.post_id, delta.in_comment, f"{comment_author} (ACTUAL) != {delta.to} (Reported)")
            # Real delta receiver: delta.to
            tobe_removed.append(i)
            while True:
                comment_id = the_comment.parent_id.split('_')[-1]
                the_comment = get_comment_by_id(delta.post_id, comment_id)
                if the_comment is None:
                    comment_id = None
                    break
                the_author = the_comment.author
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


# Constructing the dataset
def get_chat_hist(post_id: str, parent_id: str):
    text, authors = [], []
    while not parent_id == f"t3_{post_id}":
        comment = get_comment_by_id(post_id, parent_id.split('_')[-1])
        text.insert(0, comment.body)
        authors.insert(0, comment.author)
        parent_id = comment.parent_id
    return text, authors

dataset = []
for i, post in tqdm(submissions_df.iterrows(), desc="Processing submissions ..."):
    comments = comments_df[comments_df['link_id'] == f"t3_{post.id}"]
    for j, comment in comments.iterrows():
        history, history_authors = get_chat_hist(post.id, comment.parent_id)
        conversation = [post.selftext, *history, comment.body]
        conversation_authors = [post.author, *history_authors, comment.author]
        delta_count = deltas_df[
            (deltas_df['post_id'] == post.id) &
            (deltas_df['in_comment'] == comment.id)]['count'].sum()
        is_op_delta = len(deltas_df[
            (deltas_df['post_id'] == post.id) &
            (deltas_df['in_comment'] == comment.id) &
            (deltas_df['from'] == 'OP')]) > 0
        # print(post.id, comment.id, delta_count, is_op_delta)
        dataset.append([
            post.id, post.title, post.author, post.url, post.ups, post.downs, post.score, post.created_utc,
            comment.id, comment.author, comment.ups, comment.downs, comment.score, comment.author_flair_text, comment.created_utc,
            conversation, conversation_authors, len(conversation),
            delta_count, is_op_delta,
        ])

pd.DataFrame.from_records(dataset,
                          columns=[
            "post_id", "post_title", "post_author", "post_url", "post_ups", "post_downs", "post_score", "post_created_utc",
            "comment_id", "comment_author", "comment_ups", "comment_downs", "comment_score", "comment_author_flair_text", "comment_created_utc",
            "conversation", "conversation_authors", "conversation_len",
            "delta_count", "is_op_delta",
                          ]).to_json(output_path,
                                     orient='records',
                                     lines=True)
