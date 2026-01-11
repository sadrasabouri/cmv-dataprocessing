"""Functions module"""

from typing import List, Tuple, Dict, Union
import re
import json
import pickle
from tqdm import tqdm
import pandas as pd
from .params import DELTA_RE, DELTA_DEFAULT, NONE_ELEMENTS
from .params import POST_FOOTNOTE, ENDING_LINE


def split_from_users(from_block: str) -> List:
    """
    Parse the user information handles 'OP', '/u/foo', '/u/foo and /u/bar', '/u/foo, /u/bar, and /u/baz'.

    :param from_block: block of the text with the from user (delta giver) information
    """
    from_block = from_block.strip()

    if from_block == "OP":
        return ["OP"]

    users = re.findall(r"/u/([A-Za-z0-9_-]+)", from_block)
    return users


def parse_deltas(text: str) -> List[List[str]]:
    """
    Parse the delta information.

    :param text: input text
    """
    rows = []

    for m in DELTA_RE.finditer(text):
        post_id = m.group("post_id")
        to_user = m.group("to")[3:]
        comment_id = m.group("comment_id")

        total_count = int(m.group("count"))

        comment_text = m.group("comment_text") or ""
        comment_prefix = re.sub(r"\s+", " ", comment_text)[:20]

        from_users = split_from_users(m.group("from_block"))

        # divide deltas evenly across givers
        per_user_count = total_count // len(from_users)

        for from_user in from_users:
            rows.append(
                [
                    post_id,
                    from_user,
                    to_user,
                    comment_id,
                    per_user_count,
                    comment_prefix,
                ]
            )

    return rows


def extract_indexed_post_map(path_to_posts: str, save: bool = True) -> Dict[str, Dict]:
    """
    Extract the mappings to posts indexed with post id.

    :param path_to_posts: path to the posts jsonl
    :param save: flag indicating saving of the files
    """
    pid2post = {}
    with open(path_to_posts, 'r') as f:
        for line in tqdm(f, desc="Processing submissions ..."):
            post = json.loads(line.strip())
            pid2post[post.get('id')] = post
    if save:
        print("Saving id indexed to submissions files (~pid2post.pkl) ...")
        with open('~pid2post.pkl', 'wb') as f:
            pickle.dump(pid2post, f)
    return pid2post


def extract_indexed_comment_map(
        path_to_comments: str,
        save: bool = True) -> Dict[str, Dict[str, Dict]]:
    """
    Extract the mappings to comments indexed with post and comment id.

    :param path_to_comments: the path to the comments jsonl
    :param save: flag indicating saving of the files
    """
    pid2cid2comment = {}
    with open(path_to_comments, 'r') as f:
        for line in tqdm(f, desc="Making comment index files ..."):
            comment = json.loads(line.strip())
            post_id = comment.get('link_id', '').split('_')[-1]
            comment_id = comment.get('id')
            if not post_id in pid2cid2comment:
                pid2cid2comment[post_id] = {}
            if not comment_id in pid2cid2comment[post_id]:
                pid2cid2comment[post_id][comment_id] = comment

    if save:
        print("Saving id indexed to comments files (~pid2cid2comment.pkl) ...")
        with open('~pid2cid2comment.pkl', 'wb') as f:
            pickle.dump(pid2cid2comment, f)
    return pid2cid2comment


def get_comment_by_id(post_id: str, comment_id: str, pid2cid2comment: Dict[str, Dict[str, Dict]]) -> Dict:
    """
    Safely return the desired comment and None if not found.
    
    :param post_id: post id
    :param comment_id: comment id
    :param pid2cid2comment: post id -> comment id -> comment mapping
    """
    if post_id not in pid2cid2comment:
        return None
    if comment_id not in pid2cid2comment[post_id]:
        return None
    return pid2cid2comment[post_id][comment_id]


def fix_deltas(
        deltas_df: pd.DataFrame,
        pid2cid2comment: Dict[str, Dict[str, Dict]],
        save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fix delta bot inconsistency issues (~66k of deltas out of 94k).
    In some cases deltalog-bot reported delta to the comment which gives the delta not the one that gets it.
    The proxy used for findings such instances is to find cases where the author of the comment is not the same as the reported author by deltalog-bot
    
    :param deltas_df: initial deltas dataframe
    :param pid_cid2comment: post id -> comment id -> comment mapping
    :param save: flag indicating saving the files
    """
    removed_ids = []
    added_items = []
    for i, delta in tqdm(deltas_df.iterrows(), desc="Fixing deltalog-bot issue ..."):
        the_comment = get_comment_by_id(delta.post_id, delta.in_comment, pid2cid2comment)
        if the_comment is None:
            continue
        comment_author = the_comment.get('author')
        if comment_author != delta.to:
            # Real delta receiver: delta.to
            removed_ids.append(i)
            while True:
                comment_id = the_comment.get('parent_id', '').split('_')[-1]
                the_comment = get_comment_by_id(delta.post_id, comment_id, pid2cid2comment)
                if the_comment is None:
                    comment_id = None
                    break
                the_author = the_comment.get('author')
                if the_author == delta.to:
                    break
            added_items.append([delta.post_id, delta['from'], delta.to, comment_id, delta['count'], None])
    removed_items = deltas_df.iloc[removed_ids].copy()
    deltas_df = deltas_df.drop(removed_ids)
    added_items = pd.DataFrame.from_records(added_items, columns=['post_id','from','to','in_comment','count','prefix'])
    deltas_df = pd.concat([deltas_df, added_items], ignore_index=True).reset_index()
    if save:
        deltas_df.to_csv('~deltas.fixed.csv', index=False)
    return deltas_df, removed_items, added_items


def extract_deltas(deltas_df: pd.DataFrame, save: bool = True) -> Dict[Tuple[str, str], Dict[str, Union[bool, int]]]:
    """
    Extract deltas dict from deltas dataframe.

    :param deltas_df: Deltas dataframe
    :parma save: flag indicating saving the files
    """
    deltas = {}
    for _, delta in tqdm(deltas_df.iterrows(), desc="Making delta index files ..."):
        post_id = delta.post_id
        comment_id = delta.in_comment
        # cases where comment is removed (~4k of deltas out of 94k)
        if post_id is None or comment_id is None:
            continue
        
        if (post_id, comment_id) not in deltas:
            deltas[(post_id, comment_id)] = DELTA_DEFAULT.copy()
        deltas[(post_id, comment_id)]["count"] += int(delta['count'])
        if delta['from'] == "OP":
            deltas[(post_id, comment_id)]["is_op_delta"] = True
    if save:
        print("Saving id indexed to comments files (~deltas_dict.pkl) ...")
        with open('~deltas_dict.pkl', 'wb') as f:
            pickle.dump(deltas, f)
    return deltas


def get_delta(
        post_id: str,
        comment_id: str,
        deltas: Dict[Tuple[str, str], Dict[str, Union[bool, int]]]) -> Dict[str, Union[bool, int]]:
    """
    Safely return the delta information for a comment.

    :param post_id: post id
    :param comment_id: comment id
    :param deltas: delta dictionary
    """
    if (post_id, comment_id) not in deltas:
        return DELTA_DEFAULT
    return deltas[(post_id, comment_id)]


def get_chat_hist(
        post_id: str,
        parent_id: str,
        pid2cid2comment: Dict[str, Dict[str, Dict]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Return the chat history in a post.
    
    :param post_id: post id
    :param parent_id: comment id
    :param pid2cid2comment: post id -> comment id -> comment mapping
    """
    text, authors, ids = [], [], []
    while not parent_id == f"t3_{post_id}":
        comment = get_comment_by_id(post_id, parent_id.split('_')[-1], pid2cid2comment)
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


def is_non(text: str) -> bool:
    """
    Determines if a string is None.

    :param text: the give text
    """
    return text in NONE_ELEMENTS


def has_non_in_conv(record: pd.Series) -> bool:
    """
    Determine if the record conversation has a none element anywhere.
        It is a strict condition; not having them increased data by
        25% but that was not a good sacrifice.

    :param record: the target record
    """
    none_in_conv = any([is_non(x) for x in record['conversation']])
    none_in_title = is_non(record['post_title'])
    return none_in_conv or none_in_title


def post_text_cleaning(text: str) -> str:
    """
    Clean the post text.

    :param text: the post text
    """
    text = re.sub(POST_FOOTNOTE, '', text)
    text = re.sub(ENDING_LINE, '', text)
    return text
