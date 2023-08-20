import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

peft_model_id = "./pretrained_lora_model"
config = PeftConfig.from_pretrained(peft_model_id)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, 
    quantization_config=nf4_config,
    device_map='cuda:0'
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)



text = """[05/31/2021 9:29 AM] Golegar#5923
@relixion727
Do you think @3tnt will nuke the server?
[05/31/2021 9:29 AM] relixion727
### Answer:"""

gen_cfg = GenerationConfig(
    do_sample=True,
    temperature=1, 
    top_k=10,
    use_cache=False,
    max_new_tokens=64,
)

prompt = f"### Question: {text}\n ### Answer:"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].cuda()
generation_output = model.generate(
    input_ids=input_ids,
    generation_config=gen_cfg,
    return_dict_in_generate=True
).sequences[0]


output = tokenizer.decode(generation_output)
print(output)
# print(output.split("### Answer:")[1].strip())
