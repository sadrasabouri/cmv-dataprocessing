import json

with open('cmv_delta.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        post_author = data['author']
        comment_map = {}
        for c in data['comments']:
            comment_map[c['id']] = c
        for c in data['comments']:
            comment_author = c['author']
            if c['body'].strip() in ['[deleted]', '[removed]']:
                continue
            if post_author == comment_author:
                target_comment_id = c['parent_id'].split('_')[1]
                if target_comment_id == data['id']: # comment to the same OP's post
                    continue
                try:
                    is_response_to_a_op_deltad = comment_map[target_comment_id]['delta']['is_op_delta']
                except KeyError:
                    print("[ERROR] KeyError:")
                    print('post_id:', data['id'])
                    print('comment_id:', target_comment_id)
                    print('comment_map:', comment_map.keys())
                    exit()
                result = {
                    'post_id': data['id'],
                    'comment_id': c['id'],
                    'is_response_to_deltad_comment': is_response_to_a_op_deltad,
                    'comment_body': c['body'],
                }
                print(json.dumps(result, ensure_ascii=False))
