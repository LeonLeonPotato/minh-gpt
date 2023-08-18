from trl import DataCollatorForCompletionOnlyLM as Collator
from transformers import AutoTokenizer
import json

data = []

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

