import pandas as pd
import re
import json
import sys

data_path = sys.argv[1]
output_path = sys.argv[2]

POST_RE = re.compile(
    r"changemyview/comments/(?P<post_id>[a-z0-9]+)/"
)

DELTA_RE = re.compile(
    r"(?P<count>\d+)\s+delta[s]?\s+from\s+(?P<from_block>.+?)\s+to\s+(?P<to>/u/\S+).*?"
    r"\[(?P<comment_text>.*?)\]\("
    r"(?:https?://(?:www\.)?reddit\.com)?/r/changemyview/comments/"
    r"(?P<post_id>[a-z0-9]+)/.*?/(?P<comment_id>[a-z0-9]+)",
    re.IGNORECASE | re.DOTALL
)

def split_from_users(from_block):
    """
    Handles:
    - OP
    - /u/foo
    - /u/foo and /u/bar
    - /u/foo, /u/bar, and /u/baz
    """
    from_block = from_block.strip()

    if from_block == "OP":
        return ["OP"]

    users = re.findall(r"/u/([A-Za-z0-9_-]+)", from_block)
    return users


def parse_deltas(text):
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

df_list = []
with open(data_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        df_list.extend(parse_deltas(data['selftext']))
df = pd.DataFrame.from_records(df_list, columns=['post_id','from','to','in_comment','count','prefix'])
df.to_csv(output_path, index=False)

# text = """"""
# print(parse_deltas(text))
