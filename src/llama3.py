# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional
import fire
from llama import Llama, Dialog
import time
from tqdm import tqdm
import os


def read_txt(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    return content


def write_into_txt(text, file_path):
    f = open(file_path, 'w', encoding='utf-8')
    f.write(text)
    f.close()


def if_not_exist_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    batch = max_batch_size
    prompt_txt = sorted(os.listdir(input_dir))
    group_prompt = []
    for i in range(0, len(prompt_txt), batch):
        group_prompt.append(prompt_txt[i:i + batch])

    group_prompt = tqdm(group_prompt)
    start = time.time()
    for idx, gp in enumerate(group_prompt):

        dialogs: List[Dialog] = [[{"role": "user", "content": read_txt(f"{input_dir}/{input_file_name}")}] for
                                 input_file_name in gp]

        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        avt = (time.time() - start) / (idx + 1)
        group_prompt.set_postfix({'average time:': '{0:1.5f}'.format(avt)})
        for dialog, result, output_file_name in zip(dialogs, results, gp):
            output_dir = "../answers_txt_MITRatio/Llama3"
            if_not_exist_make_dir(output_dir)
            write_into_txt(result['generation']['content'], f"{output_dir}/{output_file_name}")


if __name__ == "__main__":
    input_dir = "../scenario_txt_Nov_MITRatio"
    fire.Fire(main)
