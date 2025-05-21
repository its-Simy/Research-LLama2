from datasets import load_dataset, Dataset
from transformers import (
    BitsAndBytesConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    IntervalStrategy
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
import math
import json
import time

# ─── 0) FLATTEN RAW DIALOGUE JSON INTO PROMPT/COMPLETION RECORDS ─────────────
raw = load_dataset("json", data_files="educational-fine-tuning-data/cleaned_textbook1_data.json")["train"]
flat = []
for rec in raw:
    system_ctx = ""
    for turn in rec["dialogue"]:
        if turn["from"] == "system":
            system_ctx = turn["value"].strip() + "\n\n"
            break
    for i in range(len(rec["dialogue"]) - 1):
        if rec["dialogue"][i]["from"] == "human" and rec["dialogue"][i+1]["from"] == "gpt":
            flat.append({
                "prompt": system_ctx + rec["dialogue"][i]["value"].strip(),
                "completion": rec["dialogue"][i+1]["value"].strip()
            })
flat_ds = Dataset.from_list(flat)

# ─── 1) SPLIT INTO TRAIN / VAL / TEST ─────────────────────────────────────────
split1   = flat_ds.train_test_split(test_size=0.1, seed=42)
train_ds = split1["train"]
temp_ds  = split1["test"]
split2   = temp_ds.train_test_split(test_size=0.5, seed=42)
val_ds   = split2["train"]
test_ds  = split2["test"]

# ─── 2) LOAD MODEL & TOKENIZER WITH GPU FP16 QUANT ─────────────────────────────
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = LlamaForCausalLM.from_pretrained(
    "/home/simy/llama2-project/llama/hf_model",
    quantization_config=quant_config,
    device_map="auto"
)
tokenizer = LlamaTokenizer.from_pretrained("/home/simy/llama2-project/llama/hf_model")
tokenizer.pad_token = tokenizer.eos_token

# ─── 3) APPLY LORA PEFT ────────────────────────────────────────────────────────
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# ─── 4) TOKENIZATION ──────────────────────────────────────────────────────────
def tokenize(batch):
    toks = tokenizer(
        batch["prompt"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    toks["labels"] = toks["input_ids"].clone()
    return toks

train_ds = train_ds.map(tokenize, batched=True, remove_columns=["prompt", "completion"])
val_ds   = val_ds.map(tokenize,   batched=True, remove_columns=["prompt", "completion"])

# ─── 5) SET DATASET FORMAT & SANITY CHECK ─────────────────────────────────────
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Check a base parameter is on GPU & in FP16
for name, param in model.base_model.named_parameters():
    if param.requires_grad and "lora" not in name:
        print(f"{name}: device={param.device}, dtype={param.dtype}")
        break

# Time one forward pass
sample = train_ds[0]
inputs = {
    "input_ids":      sample["input_ids"].unsqueeze(0).to(model.device),
    "attention_mask": sample["attention_mask"].unsqueeze(0).to(model.device),
}
with torch.no_grad(): _ = model(**inputs)  # warm-up
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(): _ = model(**inputs)
torch.cuda.synchronize(); t1 = time.time()
print(f"Single forward pass: {t1-t0:.3f} s")

# ─── 6) DETERMINE STEPS & SET TRAINING ARGS ────────────────────────────────────
train_batch_size = 4
steps_per_epoch  = math.ceil(len(train_ds) / train_batch_size)
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=train_batch_size,
    num_train_epochs=40,
    logging_steps=10,
    eval_steps=steps_per_epoch,
    save_steps=steps_per_epoch,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="none"
)
args.eval_strategy = IntervalStrategy.STEPS
args.save_strategy = IntervalStrategy.STEPS

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ─── 7) TRAIN & FINAL EVAL ────────────────────────────────────────────────────
trainer.train()
metrics = trainer.evaluate()
print(f"\nFinal validation metrics:\n{metrics}")
perplexity = math.exp(metrics["eval_loss"])
print(f"Perplexity: {perplexity:.2f}")
