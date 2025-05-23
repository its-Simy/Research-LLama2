# 🧠 Fine-Tuning LLaMA 2 for Java Tutoring

This project demonstrates how to fine-tune a LLaMA 2 language model to serve as a Java tutoring assistant using a structured, dialogue-style dataset.

We provide all necessary scripts for training, test set extraction, and reproduction of our experiment on **Ubuntu Linux** using **Python, PyTorch, and CUDA**.

---

## 🗂 Project Structure
```
.
├── .github/ISSUE_TEMPLATE/ # Issue templates (from original LLaMA repo)
├── llama/ # Cloned Meta LLaMA repo
│ ├── download.sh # Downloads model weights
│ ├── example_chat_completion.py # Optional demo script
│ └── ... # Additional Meta scripts and modules
├── educational-fine-tuning-data/ # Your structured dataset
│ ├── training.json
│ └── testing_structured_units.json
├── qlora_finetune.py # Fine-tuning script (QLoRA, early stopping)
├── separate_testing_data.py # Extracts reproducible test split
├── result.py # Runs inference on structured test units
├── requirements.txt # Python dependency list
├── setup.py # Setup file for local module install
├── README.md # 📄 This file
├── Responsible-Use-Guide.pdf # From Meta – ethical use guidelines
├── LICENSE # Meta’s model license
├── MODEL_CARD.md # Metadata about LLaMA 2 model
├── UPDATES.md # Version updates from Meta
└── USE_POLICY.md # Meta's acceptable use policy
```


---

## 🖥️ System Requirements

- Linux (tested on **Ubuntu 22.04**)
- Python 3.8+
- Git
- PyTorch with CUDA (GPU) or CPU version
- `wget`, `md5sum`
- (Optional) `conda` or `venv` for virtual environments

---

## ⚙️ Setup Guide

> 📌 If you want more step-by-step system-level instructions (e.g., installing drivers, CUDA toolkit, etc.), see our companion document:  
> `Guide to Installing LLaMA Language Model on Linux Ubuntu`  
> ⚠️ *Note: This guide includes some system-specific details from our own machine (`simy` user), so you may need to adapt paths and usernames to fit your setup.*

---

### ✅ Step 1: Clone & Install LLaMA

```bash
git clone https://github.com/facebookresearch/llama.git
cd llama
pip install -e .
```

### ✅ Step 2: Download LLaMA 2 Model Weights
Request access from Meta AI: LLaMA Downloads

Once approved, you’ll receive a signed URL.

Run the download script:

```
chmod +x download.sh
./download.sh
```
Enter the signed URL when prompted. The downloaded model (e.g., llama-2-7b-chat/) should be placed in your working directory.

### ✅ Step 3: Fine-Tuning
```
torchrun --nproc_per_node 1 qlora_finetune.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --batch_size 4
```
qlora_finetune.py is our modified training script using QLoRA with early stopping.

Expects dataset files in educational-fine-tuning-data/.

### ✅ Step 4: Recreate the Test Set
To reproduce our experimental test split:

```
python separate_testing_data.py
```
This generates testing_structured_units.json using the same data split logic from training.

### ✅ Step 5: Test Inference (Optional)
You can run model inference on the test data using:

```
python result.py
```
Ensure paths in the script match your directory layout and model files.


> ⚠️ **Note on Machine-Specific Settings:**  
> The `result.py` script includes default environment variables to ensure the script runs correctly on a single machine using CPU or a single GPU:
>
> ```python
> os.environ.setdefault("RANK", "0")
> os.environ.setdefault("WORLD_SIZE", "1")
> os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
> os.environ.setdefault("MASTER_PORT", "29500")
> ```
>
> These are standard for local execution. If you run inference in a multi-GPU or distributed setting, you may need to override these with your own cluster's settings or use a launcher like `torchrun`.

### 🧪 Dataset Format
The data used for the model came from this repository:
https://github.com/its-Simy/educational-fine-tuning-data

Each training or testing sample is stored as a multi-turn dialogue JSON object like this:
```
{
  "dialogue": [
    {
      "from": "system",
      "value": "You are a beginner-friendly Java professor. Respond to each student question in 3–4 clear sentences. Do not include Java code, and only use analogies when they clarify input-related or abstract behavior."
    },
    {
      "from": "human",
      "value": "How do you instantiate a Scanner object to read from the keyboard?"
    },
    {
      "from": "gpt",
      "value": "To read keyboard input, you create a Scanner object connected to the standard input stream..."
    },
    {
      "from": "human",
      "value": "How do you read an integer from the user using Scanner?"
    },
    {
      "from": "gpt",
      "value": "You use a method on the Scanner object that waits for and returns the next integer entered by the user..."
    },
    {
      "from": "gpt",
      "value": "What happens when Scanner’s nextInt() is called but the user types a word instead?\nA) It accepts the input as a string\nB) It skips the word and moves on\nC) It throws an input mismatch error\nD) It converts the word to zero"
    },
    {
      "from": "human",
      "value": "C"
    },
    {
      "from": "gpt",
      "value": "Correct! If the input doesn’t match the expected data type, Scanner throws an input mismatch exception..."
    }
  ]
}
```
Each object in the dataset contains a dialogue array.

Turns alternate between human and gpt, optionally starting with a system message.

Some sequences contain quiz-like questions followed by an answer and explanation.
We used a format that simulates tutoring conversations and structured learning flows. Testing samples include multiple-choice practice questions following several explanation turns.

### 📄 License & Usage
This project uses Meta’s LLaMA 2 models under their official license agreement. You must accept their terms to download and use the model weights.

### 📚 References
Meta LLaMA 2 Research Paper

LLaMA Download Portal

### 🙋 Support
For questions about the setup or reproducing our experiment, feel free to open an issue or reach out.
Contact me via email: [Ramis50@farmingdale.edu](mailto:Ramis50@farmingdale.edu)

---

Let me know if you'd like:
- A companion `Guide-to-Installing.txt` file for that Ubuntu setup guide.
- A GitHub Actions CI setup to verify your scripts work.
- A badge/banner to make the repo look more polished.

I can also generate the Ubuntu setup file in `.md` or `.txt` format if you'd like it separate and cleaner.
