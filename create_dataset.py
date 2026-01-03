from typing import List, Dict
import json
import sys

data_path = sys.argv[1]

def get_chat_hist(post: Dict, comment: Dict, id2index: Dict):
    text, authors = [], []
    while not comment['parent_id'] == f"t3_{post['id']}":
        comment = post['comments'][id2index[comment['parent_id']]]
        text.insert(0, comment['body'])
        authors.insert(0, comment['author'])
    return text, authors
        

with open(data_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        post = json.loads(line)
        comment_id2index = {f"t1_{c['id']}": i for i, c in enumerate(post['comments'])}
        for comment in post['comments']:
            history_text, comment_history_author = get_chat_hist(post, comment, comment_id2index)
            result = {
                    'post_id': post.get('id'),
                    'post_title': post.get('title'),
                    'post_author': post.get('author'),
                    'post_text': post.get('selftext'),
                    'post_url': post.get('url'),
                    'post_ups': post.get('ups'),
                    'post_downs': post.get('downs'),
                    'post_score': post.get('score'),
                    'post_created_utc': post.get('created_utc'),
                    ############
                    'comment_id': comment.get('id'),
                    'comment_text': comment.get('body'),
                    'comment_history_text': history_text,
                    'comment_history_author': comment_history_author,
                    'comment_author': comment.get('author'),
                    'comment_ups': comment.get('ups'),
                    'comment_downs': comment.get('downs'),
                    'comment_score': comment.get('score'),
                    'comment_author_flair_text': comment.get('author_flair_text'),
                    'comment_delta_count': comment.get('delta', {}).get('count'),
                    'comment_delta_is_op': comment.get('delta', {}).get('is_op_delta'),
                    'comment_created_utc': comment.get('created_utc'),
                }
            print(json.dumps(result, ensure_ascii=False))
