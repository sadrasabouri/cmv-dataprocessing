import praw
from tqdm import tqdm

reddit = praw.Reddit(
    client_id="786O1er4gJS1dpSiIdvBWA",
    client_secret="_8qkSYxJDqt5ZBlY-ub-45ito_EZBw",
    user_agent="ChangeMyViewScraper"
)

subreddit = reddit.subreddit("cmv")

for post in subreddit.new(limit=None):  # you can use .hot(), .top(), etc.
    print(post.title, post.url)
