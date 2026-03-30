import json
import mailbox
import re
from email.utils import parsedate_to_datetime
from email.message import Message
from mailbox import mboxMessage
from os import makedirs
from os.path import join, dirname

import langdetect

CLEAN_EMAILS = []
DISCARDED_EMAILS = {
    "non_english": [],
    "forwarded": [],
    "short": [],
    "empty": [],
    "cant_decode_utf8": [],
}

SHORT_EMAIL_THRESHOLD = 10  # words

def count_words(s):
    return len(s.split())


def filter_snippet(msg):
    try:
        plain_text = msg["snippet_text"]
    except:
        DISCARDED_EMAILS["cant_decode_utf8"].append(msg)
        return None

    if plain_text is None:
        return None


    # check length before detecting language
    if count_words(plain_text) < SHORT_EMAIL_THRESHOLD:
        DISCARDED_EMAILS["short"].append(plain_text)
        return None
    try:
        if langdetect.detect(plain_text) != "en":
            DISCARDED_EMAILS["non_english"].append(msg)
            return None
    except:
        # failed to detect language
        DISCARDED_EMAILS["non_english"].append(msg)
        return None

    if plain_text.isspace() or plain_text == "":
        DISCARDED_EMAILS["empty"].append(msg)
        return None

    msg["snippet_text"] = plain_text.strip()
    return msg


def process_snippet(snippet):
    assert 'source_path' in snippet
    assert 'snippet_text' in snippet

    return snippet

def accept_snippet(snippet):
    if filter_snippet(snippet):
        return True
    return False


def extract_snippets(snippets_path, output_path, save_discarded_snippets_path):


    with open(snippets_path, 'r') as f:
        snippets_pre = json.load(f)

    processed_snippets = [filter_snippet(snippet) for snippet in snippets_pre]

    accepted_snippets = [snippet for snippet in processed_snippets if snippet is not None]

    


    makedirs(dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in accepted_snippets:
            json_record = json.dumps(item)
            f.write(json_record + "\n")

    if sum([len(v) for v in DISCARDED_EMAILS.values()]) > 0:
        makedirs(dirname(save_discarded_snippets_path), exist_ok=True)
        with open(save_discarded_snippets_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(DISCARDED_EMAILS))
