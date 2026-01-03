import requests
import time

subreddit = "cmv"
url = "https://api.pushshift.io/reddit/submission/search/"
params = {"subreddit": subreddit, "size": 100, "sort": "desc", "sort_type": "created_utc"}

all_posts = []
while True:
    r = requests.get(url, params=params)
    data = r.json().get("data", [])
    if not data:
        break
    all_posts.extend(data)
    params["before"] = data[-1]["created_utc"]  # move to older posts
    print(f"Collected {len(all_posts)} posts so far...")
    time.sleep(1)  # be nice to the API

print("Total posts collected:", len(all_posts))
