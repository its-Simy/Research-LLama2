# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: int = 256,
    max_batch_size: int = 4,
):
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
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    #user_prompt= input("What do you want to learn about: ")

    system_prompt = (
    "You are a helpful and intelligent Java professor for an introductory course. "
    "Answer **directly**, in **no more than 3–4 sentences**, **without any greetings** or closings, **don't repeat back the question** "
    "Explain concepts clearly and concisely, using simple language and a short real‑world analogy. Do not provide code."
    )


    question_count = 0
    max_questions  = 3

    while True:
        user_input = input("🔹 Student: ").strip()
        if not user_input:
            break

        #checks to see if it has reached the threshold of questions before the print
        if question_count < max_questions:

            # build your chat-formatted prompt
            chat_input = (
            "<s>[INST] "
            + system_prompt
            + " [/INST]\n\n"
            + user_input
            + "\n\n[INST]"
            )

            # generate and print reply
            results = generator.text_completion(
            prompts=[chat_input],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            )
            #[0]["generation"].strip()
            #print(f"\n💡 Professor: {result}\n")

            print(f"\n Professor: {results[0]['generation'].strip()} \n")
            question_count += 1

        
        if question_count == max_questions:
            # build your chat-formatted prompt
            chat_input = (
            "<s>[INST] "
            + system_prompt
            + " [/INST]\n\n"
            + user_input
            + "\n\n[INST]"
            )

            # generate and print reply
            results = generator.text_completion(
            prompts=[chat_input],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            )
            print(f"\n Professor: {results[0]['generation'].strip()} \n")
            print("\n🚀 You’ve asked 3 questions—thanks! Exiting now.")
            break
        '''
        else:
            # after 3 questions
            print("🚀 You’ve asked 3 questions—thanks! Exiting now.")
            break
        '''

   
       


if __name__ == "__main__":
    fire.Fire(main)

    

    