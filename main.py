import json
from model import model, tokenizer
from utils import print_trainable_parameters
from datasets import Dataset
import threading
import torch
import time

from transformers.training_args import OptimizerNames
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["v_proj", "q_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

# Prepare model, peft stuff I don't really understand
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)

print_trainable_parameters(model) # Debugging

# Loading dataset
data = []
with open("dataset.jsonl") as f:
    for line in f.readlines():
        data.append(json.loads(line))

data = Dataset.from_list(data).shuffle()

request_template = "### Question:"
response_template = "### Answer:"
collator = DataCollatorForCompletionOnlyLM(
    response_template, 
    request_template, 
    tokenizer=tokenizer, 
    mlm=False
)

# Training arguments
output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
)

# ??? IDK what this does
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"{request_template} {example['prompt'][i]}\n {response_template} {example['completion'][i]}"
        output_texts.append(text)
    return output_texts

# Training object
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    peft_config=config,
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
# train_thread = threading.Thread(target=trainer.train, daemon=True)
# train_thread.start()
# while True:
#     time.sleep(3)
trainer.train()