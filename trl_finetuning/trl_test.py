# 0. imports
import torch
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, PreTrainedModelWrapper

# 1. load a pretrained model
# model = AutoModelForCausalLMWithValueHead.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
# ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
# tokenizer = GPT2Tokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
# tokenizer.pad_token = tokenizer.eos_token

model_name = "tencent/Hunyuan-1.8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
ref_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
model = AutoModelForCausalLMWithValueHead(model)
ref_model = AutoModelForCausalLMWithValueHead(ref_model)

# 2. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# 3. encode a query
query_txt = "# here is an example of random 10x10 binary grid and nothing else: 0100110101\n0101010001\n1010100101\n01010"
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 100,
}
response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=False, **generation_kwargs)
print(response_tensor)
response_txt = tokenizer.decode(response_tensor[0])
print(response_txt)

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
# def reward(s):
#     if len(s) >= 25*25: return 1.0
#     else: return -10.0

# reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# # 6. train model with ppo
# train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
