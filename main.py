import json

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from model import model, tokenizer
from utils import print_trainable_parameters


# Set up a LoraConfig
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

request_template = "### Question:\n"
response_template = "### Answer:\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template, 
    request_template, 
    tokenizer=tokenizer, 
    mlm=False
)

# Training arguments
training_arguments = TrainingArguments(
    output_dir = "./results",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    optim = "paged_adamw_32bit",
    logging_steps = 10,
    learning_rate = 0.0002,
    bf16 = True,
    max_grad_norm = 0.35,
    num_train_epochs = 1
)

# ??? IDK what this does
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"{request_template}\n{example['prompt'][i]}\n{response_template}\n{example['completion'][i]}"
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