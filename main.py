import json

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from model import model, tokenizer
from utils import *

# For ease of debugging
print_gpu_utilization()

# Set up a LoraConfig
config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["v_proj", "q_proj", "lm_head"], 
    # target_modules=["c_fc", "c_attn"], 
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
training_arguments = TrainingArguments(
    output_dir = "./output",
    per_device_train_batch_size = 32,
    gradient_accumulation_steps = 4,
    optim = "paged_adamw_32bit",
    logging_steps = 1,
    learning_rate = 0.00035,
    bf16 = True,
    max_grad_norm = 0.35,
    num_train_epochs = 2,
    warmup_steps = 15,
    lr_scheduler_type = "constant_with_warmup",
    report_to = "wandb"
)

# ??? IDK what this does
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        # text = f"{request_template}\n{example['prompt'][i]}\n{response_template}\n{example['completion'][i]}"
        text = f"{request_template} {example['prompt'][i]}\n {response_template} {example['completion'][i]}"
        output_texts.append(text)
    return output_texts

# Training object
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    peft_config=config,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.save_pretrained("./output")
model.push_to_hub(
    "3tnt/minh-gpt",
    commit_message="First commit",
    max_shard_size="4GB",
    create_pr=1
    # token = getpass.getpass("Enter your HuggingFace token (write must be enabled!): ")
)