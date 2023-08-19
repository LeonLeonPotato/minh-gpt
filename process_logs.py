import re
import json
import pandas as pd
from collections import deque
from transformers import AutoTokenizer
import bisect

model_id = "meta-llama/Llama-2-7b-hf"
TIMESTAMP = r"^\[\d{2}\/\d{2}\/\d{4}\ \d{1,2}:\d{2}\ (?:AM|PM)\]\ .+$"
MAX_TOKENS = 512 - 64 # Leave some space for the answer
MAX_LINES = 128 # We don't want too many lines of context... right?
LINK_PATTERN = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
LINKS = [
    "images-ext-1.discordapp.net",
    "images-ext-2.discordapp.net",
    "images-ext-3.discordapp.net",
    "images-ext-4.discordapp.net",
    "cdn.discordapp.com",
    "tenor.com",
    "giphy.com",
    "media.tenor.com",
    "media.giphy.com",
    "i.imgur.com"
]
EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".mp4",
    ".webm"
]

with open('shitverse.txt') as f:
    data = f.readlines()
    data = [d.strip() for d in data]
    data = [d for d in data if len(d) > 0]

tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

cur_speaker = None
cur_date = 0
relixion_logs = []
other_logs = deque([""], maxlen = MAX_LINES)
other_accumulator = deque([0], maxlen = MAX_LINES)

prompts, completions = [], []
_lens = []

def count_tokens(st):
    return len(tokenizer(st)['input_ids'])

def add_line(st):
    other_logs.append(st)
    other_accumulator.append(count_tokens(st) + other_accumulator[-1])

def process_links(st):
    def process_link(link):
        link = link.group(0)
        _, __, host, *path = link.split('/')
        if host not in LINKS: return link
        file = path[-1]
        if not any([file.endswith(ext) for ext in EXTENSIONS]):
            file += ".unknown"
        return f"[File: {file}]"
        
    return re.sub(LINK_PATTERN, process_link, st)

for linenum, line in enumerate(data):
    matched_pattern = re.findall(TIMESTAMP, line)
    if matched_pattern:
        # magic??
        cur_speaker = matched_pattern[0][1:].split(' ')[3:][0]

    line = process_links(line)

    if cur_speaker == "relixion727":
        relixion_logs.append(line)
    else:
        if relixion_logs:
            add_line(relixion_logs[0])

            idx = bisect.bisect_left(
                other_accumulator, 
                other_accumulator[-1] - MAX_TOKENS,
                lo = 0,
                hi = len(other_accumulator) - 1
            ) + 1

            prompt = [other_logs[i] for i in range(idx, len(other_logs))]
            prompt = "\n".join(prompt)
            _lens.append(count_tokens(prompt))
            prompts.append(prompt)

            completion = ""
            for relixion_line in relixion_logs[1:]:
                add_line(relixion_line)
                completion += relixion_line + "\n"
            
            completions.append(completion[:-1])

            relixion_logs = []

        add_line(line)

print(pd.DataFrame(_lens).describe())

with open("dataset.jsonl", "w", encoding='utf-8') as f:
    for p, c in zip(prompts, completions):
        st = json.dumps({"prompt": p, "completion": c})
        f.write(st + "\n")