from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

model_name = "gpt2"
dataset = "./random_grids2.json"

def formatting_func(example):
    return f"{example['prompt']}\n{example['completion']}"

dataset = load_dataset("json", data_files=dataset, split="train")

dataset = dataset.map(lambda x: {"text": formatting_func(x)})

sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=1024,
    output_dir="./gpt2_finetuned",

    # Logging
    logging_steps=10,              # alle 10 Schritte loggen
    logging_dir="./gpt2_logs",          # optional: wohin Logs geschrieben werden

    # Checkpoints
    save_steps=100,                # alle 100 Schritte Checkpoint speichern
    save_total_limit=2,            # maximal 2 Checkpoints behalten

    # Training
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    fp16=True,                     # falls GPU vorhanden
)


trainer = SFTTrainer(
    model_name,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()