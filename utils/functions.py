"""Functions module"""

from typing import List
import re
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
