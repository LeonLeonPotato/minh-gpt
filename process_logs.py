import re
import json
from datetime import datetime
from collections import deque
from transformers import AutoTokenizer
import bisect

model_id = "meta-llama/Llama-2-7b-hf"
TIMESTAMP = r"^\[\d{2}\/\d{2}\/\d{4}\ \d{1,2}:\d{2}\ (?:AM|PM)\]\ .+$"
MAX_TOKENS = 1024 - 256 # Leave some space for the answer
MAX_LINES = 64 # We don't want too many lines of context... right?

with open('shitverse.txt') as f:
    data = f.readlines()
    data = [d.strip() for d in data]
    data = [d for d in data if len(d) > 0]

tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

cur_speaker, last_speaker = None, None
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

for linenum, line in enumerate(data):
    matched_pattern = re.findall(TIMESTAMP, line)
    if matched_pattern:
        matched_pattern = matched_pattern[0]
        # Example pattern: [10/02/2020 11:34 PM] relixion727
        tmp = matched_pattern[1:].split(' ')
        date, time, ampm, cur_speaker = tmp[0], tmp[1], tmp[2][:-1], tmp[3:][0]
        cur_date = datetime.strptime(f"{date} {time} {ampm}", "%m/%d/%Y %I:%M %p").timestamp()
        del date, time, ampm, tmp

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
            prompt = "\n".join(prompt) + "\n"
            _lens.append(count_tokens(prompt))
            prompts.append(prompt)

            completion = ""
            for relixion_line in relixion_logs[1:]:
                add_line(relixion_line)
                completion += relixion_line + "\n"
            
            completions.append(completion[:-1])

            relixion_logs = []

        add_line(line)

    last_speaker = cur_speaker

import pandas as pd
import numpy as np

df_describe = pd.DataFrame(np.array(_lens))
print(df_describe.describe())

with open("dataset.jsonl", "w", encoding='utf-8') as f:
    for p, c in zip(prompts[:500], completions[:500]):
        st = json.dumps({"prompt": p, "completion": c})
        f.write(st + "\n")