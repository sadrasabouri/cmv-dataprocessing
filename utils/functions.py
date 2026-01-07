"""Functions module"""

from typing import List, Tuple, Dict
import re
import json
from tqdm import tqdm
import pickle
from .params import DELTA_RE


def split_from_users(from_block: str) -> List:
    """
    Parse the user information handles 'OP', '/u/foo', '/u/foo and /u/bar', '/u/foo, /u/bar, and /u/baz'.

    :param from_block: block of the text with the from user (delta giver) information.
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


def get_indexed_comment_maps(path_to_comments: str, save: bool = True) -> Tuple[Dict[Tuple[str, str], Dict], Dict[str, List[Dict]]]:
    """
    Extract the mappings to comments indexed with post and comment id.

    :param path_to_comments: the path to the comments jsonl.
    :param save: flag indicating saving of the files.
    """
    # TODO: make it a pid: cid: comment map
    pid_cid2comment = {}
    pid2comment = {}
    with open(path_to_comments, 'r') as f:
        for line in tqdm(f, desc="Making comment index files ..."):
            comment = json.loads(line.strip())
            post_id = comment.get('link_id', '').split('_')[-1]
            comment_id = comment.get('id')
            pid_cid2comment[(post_id, comment_id)] = comment

            if not post_id in pid2comment:
                pid2comment[post_id] = []
            pid2comment[post_id].append(comment)
    if save:
        print("Saving id indexed to comments files (~pid_cid2comment.pkl) ...")
        with open('~pid_cid2comment.pkl', 'wb') as f:
            pickle.dump(pid_cid2comment, f)
        print("Saving id indexed to comments files (~pid2comment.pkl) ...")
        with open('~pid2comment.pkl', 'wb') as f:
            pickle.dump(pid2comment, f)
    return pid_cid2comment, pid2comment
