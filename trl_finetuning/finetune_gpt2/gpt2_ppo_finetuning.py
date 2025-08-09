from transformers import AutoTokenizer, GPT2TokenizerFast
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import torch
import re
import os

# Achtung: ich benutze wie immer python==3.8 und deswegen ist trl==0.11.4

# === GPU check ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Modellpfad ===
MODEL_PATH = "./gpt2_finetuned/checkpoint-1500"

# === Tokenizer & Modell laden ===
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token

# Modell mit Value Head laden und direkt auf GPU verschieben
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_PATH).to(device)

# === PPO-Konfiguration ===
config = PPOConfig(
    model_name=MODEL_PATH,
    batch_size=1,
    mini_batch_size=1,
    optimize_cuda_cache=True
)

# === PPO-Trainer (nutzt automatisch GPU via Accelerate) ===
ppo_trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)

# === Fester Prompt ===
prompt = "Create a 10x10 maze by returning a list of 2d coordinates. " \
"This list indicates where we want to place walls. " \
"Also specify a starting position and a goal via coordinates." \
"Place the walls such that the shortest path between start and goal is long."

# === Reward-Funktion ===
def reward_func(text):
    # Finde alle Tupel der Form (x,y), wobei x und y Ziffern 0–9 sind
    matches = re.findall(r'\((\d),\s*(\d)\)', text)
    reward = 0.0
    if 0 <= len(matches) and len(matches) <= 102:
        reward += 2.0
    else: 
        reward -= 2.0
    
    # finde "goal: (x,y)" 
    matches = re.findall(r'goal:\s*\((\d),\s*(\d)\)', text)
    if len(matches) == 1: reward += 5.0
    else: reward -= 5.0

    # finde "start: (x,y)"
    matches = re.findall(r'start:\s*\((\d),\s*(\d)\)', text)
    if len(matches) == 1: reward += 5.0
    else: reward -= 5.0

    # bestrafe zu lange Ausgaben: 
    if len(text) >= 500: reward -= len(text)/100
    
    return reward


# === Prompt vorbereiten (einmalig) ===
prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True)
query_tensor = prompt_tokens["input_ids"].to(device)
query_attention_mask = prompt_tokens["attention_mask"].to(device)

# === Speichereinstellungen ===
SAVE_PATH = "./ppo_finetuned_gpt2/"
SAVE_EVERY = 20

# === PPO Training Loop ===
for step in tqdm(range(100), desc="PPO Training"):
    # Antwort generieren
    response_ids = model.generate(
        input_ids=query_tensor,
        attention_mask=query_attention_mask,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    # Nur den neu generierten Teil extrahieren
    gen_segment = response_ids[0][query_tensor.shape[-1]:]
    generated_text = tokenizer.decode(gen_segment, skip_special_tokens=True)

    # Reward berechnen
    score = float(reward_func(generated_text))

    # Response in Batch-Form bringen
    response_tensor = gen_segment.unsqueeze(0)  # Shape: (1, seq_len)

    # PPO Schritt – alles als Tensoren übergeben
    ppo_trainer.step(
        [query_tensor.squeeze(0)],     # queries
        [response_tensor.squeeze(0)],  # responses
        [torch.tensor(score, device=device)]                         # score
    )

    # Logging
    tqdm.write(f"[{step}] Score: {score:.2f}") #| Output: {generated_text}

    # Speichern alle N Schritte
    if step > 0 and step % SAVE_EVERY == 0:
        save_dir = f"{SAVE_PATH}checkpoint-step{step}"
        os.makedirs(save_dir, exist_ok=True)
        ppo_trainer.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        tqdm.write(f"💾 Modell gespeichert in: {save_dir}")

# Endgültiges Modell speichern
SAVE_PATH += "END"
os.makedirs(SAVE_PATH, exist_ok=True)
ppo_trainer.model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
tqdm.write(f"✅ Endmodell gespeichert in: {SAVE_PATH}")
