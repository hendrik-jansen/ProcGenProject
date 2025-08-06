from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

model_name = "google/gemma-2b"
dataset_path = "./random_grids2.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Formatierungsfunktion
def formatting_func(example):
    return f"{example['prompt']}\n{example['completion']}"

print(1)
# Dataset laden und vorbereiten
dataset = load_dataset("json", data_files=dataset_path, split="train")
print(2)
dataset = dataset.map(lambda x: {"text": formatting_func(x)})
print(3)

# DeepSpeed Konfiguration (Stage 2 mit offload kann Speicher stark reduzieren)
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",  # Entlastet GPU RAM
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "fp16": {
        "enabled": True
    },
    "steps_per_print": 1,
    "wall_clock_breakdown": False
}

# SFT-Konfiguration inkl. DeepSpeed
sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=2048,
    output_dir="./gemma_finetuned",
    
    # Logging
    logging_steps=1,
    logging_dir="./gemma_logs",

    # Checkpoints
    save_steps=100,
    save_total_limit=2,

    # Training
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    fp16=True,

    # DeepSpeed aktivieren
    deepspeed=deepspeed_config
)

# Trainer initialisieren
trainer = SFTTrainer(
    model=model_name,
    train_dataset=dataset,
    args=sft_config,
)

# CUDA-Speicherinfo (optional)
print(torch.cuda.memory_summary())

# Training starten
trainer.train()
