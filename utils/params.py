"""Parameter module"""

import re


# Data-driven regex that is tested to capture all different cases
DELTA_RE = re.compile(
    r"(?P<count>\d+)\s+delta[s]?\s+from\s+(?P<from_block>.+?)\s+to\s+(?P<to>/u/\S+).*?"
    r"\[(?P<comment_text>.*?)\]\("
    r"(?:https?://(?:www\.)?reddit\.com)?/r/changemyview/comments/"
    r"(?P<post_id>[a-z0-9]+)/.*?/(?P<comment_id>[a-z0-9]+)",
    re.IGNORECASE | re.DOTALL
)
