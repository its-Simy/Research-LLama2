# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""
Entry point of the program for generating text using a pretrained model.
Args:
    ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
    tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
    temperature (float, optional): The temperature value for controlling randomness in generation.
        Defaults to 0.6.
    top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
        Defaults to 0.9.
    max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
    max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
    max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
""" 

import os
# ————————————————————————————————————————————————
# Default single‑GPU/CPU distributed settings
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
# ————————————————————————————————————————————————

import fire
from llama import Llama
from typing import List


def main(
    ckpt_dir: str = "./llama-2-7b-chat-hf",
    tokenizer_path: str = "./llama-2-7b-chat-hf/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_gen_len: int = 256,
    max_batch_size: int = 4,
):
    """
    Interactive Java tutor: answers up to 3 questions, then
    generates a multiple‑choice practice question based on the discussion.
    """
    # 1) Build the chat‑tuned generator
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # 2) Define the system persona
    system_prompt = (
        "You are a helpful and intelligent Java professor for an introductory course. "
        "Answer directly, in **no more than 3–4 sentences**, **without any greetings or closings**, "
        "and don’t repeat back the question. Explain concepts clearly and concisely, "
        "using simple language and a short real‑world analogy. Do not provide code."
    )

    # 3) Initialize chat history with system prompt
    chat_history: List[dict] = [
        {"role": "system", "content": system_prompt}
    ]

    question_count = 0
    max_questions = 3

    # 4) Interactive Q&A loop
    while True:
        user_input = input("🔹 Student: ").strip()
        if not user_input:
            print("Goodbye!")
            break

        # record student turn
        chat_history.append({"role": "user", "content": user_input})

        # model generates answer based on full history
        result = generator.chat_completion(
            [chat_history],  # wrap in list for single-dialogue
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]["generation"]["content"].strip()

        # print and record professor’s reply
        print(f"\n💡 Professor: {result}\n")
        chat_history.append({"role": "assistant", "content": result})
        question_count += 1

        # after 3 exchanges, ask a practice question and exit
        if question_count >= max_questions:
            quiz_prompt = {
                "role": "user",
                "content": (
                    "📝 Practice Question:\n"
                    "Generate a single multiple‑choice question about one of the topics we just covered. "
                    "Provide exactly 4 options labeled A–D, each on its own line. Then insert two blank lines and "
                    "write:\n\nAnswer: <letter>\nExplanation: <brief explanation>"
                )
            }
            chat_history.append(quiz_prompt)

            quiz = generator.chat_completion(
                [chat_history],
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )[0]["generation"]["content"].strip()
            # ── POST‑PROCESS TO ADD 5 BLANK LINES BEFORE “Answer:” ──
            if "Answer:" in quiz:
                before, after = quiz.split("Answer:", 1)
                # ensure exactly 5 blank lines
                quiz = before.rstrip() + "\n" * 6 + "Answer:" + after.strip()

            print(f"\n📝 Quiz Time!\n\n{quiz}\n")
            break



if __name__ == "__main__":
    fire.Fire(main)

    

    