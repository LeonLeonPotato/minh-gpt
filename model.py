import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer
)

model_id = "meta-llama/Llama-2-7b-hf"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16,
   bnb_4bit_use_double_quant=True
)

model : AutoModel = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=nf4_config
)

tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token