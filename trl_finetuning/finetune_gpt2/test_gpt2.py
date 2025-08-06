from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

checkpoint_path = "./gpt2_finetuned/checkpoint-1500" # use "gpt2" to compare it to the model before finetuning
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
print("ja")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.pad_token = tokenizer.eos_token  # nötig für GPT-2

prompt = "Create a maze by returning a list of 2d coordinates. This list indicates where we want to place walls. Also specify a starting position and a goal via coordinates."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

max_total_len = 1024 
input_len = inputs["input_ids"].shape[1]
max_gen_len = max_total_len - input_len

start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_gen_len,
        do_sample=False,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
end = time.time()

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"Das Generieren hat {end - start} sek gedauert!")
