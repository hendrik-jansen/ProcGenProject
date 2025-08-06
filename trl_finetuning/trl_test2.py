from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen1.5"  # oder z. B. "gpt2", "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

prompt = "Create a binary string of length 625. Here is an example: 01010111001010111001010100101010...\n"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

output = model.generate(input_ids, max_length=64, do_sample=True, temperature=0.7)
print(tokenizer.decode(output[0]))
