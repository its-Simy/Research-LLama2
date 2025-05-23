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
    max_seq_len: int = 4096,
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

        chat_history.append({"role": "user", "content": user_input})
        result = generator.chat_completion(
            [chat_history],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]["generation"]["content"].strip()

        print(f"\n💡 Professor: {result}\n")
        chat_history.append({"role": "assistant", "content": result})
        question_count += 1

        # ── After 3 Q&A turns, generate the MCQ ──
                # ── After 3 Q&A turns, generate the MCQ ──
        if question_count >= max_questions:
            # 1) Ask for MCQ + answer key
            quiz_prompt = {
                "role": "user",
                "content": (
                    "📝 Practice Question:\n"
                    "Generate exactly one multiple-choice question on a topic we covered, "
                    "with 4 options labeled A–D (each on its own line). **AFTER** the choices, "
                    "on a new line write `Answer: <letter>` and on the next line `Explanation: <brief explanation>`. "
                    "We'll only show the question and choices to the student."
                )
            }
            chat_history.append(quiz_prompt)

            # 2) Retrieve the raw quiz (with key & explanation)
            raw_quiz = generator.chat_completion(
                [chat_history],
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )[0]["generation"]["content"].strip()

            # 3) Parse out the student-facing question & the answer key
            parts = raw_quiz.split("Answer:")
            question_and_choices = parts[0].strip()
            key_and_expl = parts[1].split("Explanation:")
            answer_key = key_and_expl[0].strip()
            explanation_key = key_and_expl[1].strip()

            # 4) Record the full raw_quiz as the assistant turn
            chat_history.append({"role": "assistant", "content": raw_quiz})

            # 5) Show only the question & choices
            print(f"\n📝 Quiz Time!\n\n{question_and_choices}\n")
            print("👉 Please respond with **only** the letter (A, B, C, or D) of your choice.")

            # 6) Read & validate the student’s choice (case‐insensitive)
            while True:
                raw = input("Your answer in lowercase: ").strip()
                if raw and raw[0].upper() in ["A", "B", "C", "D"]:
                    student_choice = raw[0].upper()
                    break
                print("Invalid choice. Please enter only A, B, C, or D.")

            # 7) Build the grading prompt with the real key
            grade_instruction = (
                f"{student_choice}\n\n"
                f"Correct answer: {answer_key}\n\n"
                "Evaluate only the student’s answer. Respond **only** with 'Correct!' or 'Incorrect.' "
                "followed by a brief explanation why it is correct or incorrect."
            )
            chat_history.append({"role": "user", "content": grade_instruction})

            # 8) Get and print the grading feedback
            grading = generator.chat_completion(
                [chat_history],
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )[0]["generation"]["content"].strip()

            print(f"\n💡 Professor: {grading}\n")
            break





if __name__ == "__main__":
    fire.Fire(main)

    

    