from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

# load & split
dataset = load_dataset("json", data_files="educational-fine-tuning-data/textbook1.json")
ds = dataset["train"].train_test_split(test_size=0.1)
train_ds = ds["test"]#Adjusted because this is not random data, so the model will in fact miss data, now training on the seperated testing data
eval_ds  = ds["test"]

model = LlamaForCausalLM.from_pretrained(
    "/home/simy/llama2-project/llama/hf_model",
    load_in_4bit=True,
    device_map="auto"
)

tokenizer = LlamaTokenizer.from_pretrained("/home/simy/llama2-project/llama/hf_model")
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

def tokenize(batch):
    output = tokenizer(
        batch["prompt"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    output["labels"] = output["input_ids"].clone()
    return output

train_ds = train_ds.map(tokenize, batched=True)
eval_ds  = eval_ds.map(tokenize, batched=True)

args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_train_epochs=3,#only for the testing training set
    logging_steps=10,
    save_strategy="epoch",
    do_eval=True,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds
)

trainer.train()
metrics = trainer.evaluate()
print(f"\nFinal evaluation metrics:\n{metrics}")