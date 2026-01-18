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

DELTA_DEFAULT = {"is_op_delta": False, "count": 0}
NONE_ELEMENTS = [None, 'null', '[deleted]', '[removed]']
POST_FOOTNOTE = r"(?:&amp;amp;gt;|&gt;|[\\>\s*])*.*?footnote.*from.*moderators?.*"
ENDING_LINE = r"\n\n_____$"

AGREEMENT_TERMS = ["I agree", "You're right", "That makes sense"]
DISAGREEMENT_TERMS = ["I disagree", "You're wrong", "That doesn't make sense"]
