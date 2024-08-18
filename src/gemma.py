# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# pip3 install transformers
# pip3 install accelerate

import os
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch


def read_txt(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    return content


def write_into_txt(text, file_path):
    f = open(file_path, 'w', encoding='utf-8')
    f.write(text)
    f.close()


login(token="")  # https://huggingface.co/settings/tokens
torch.cuda.empty_cache()

cuda_id = 0

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", device_map=f"cuda:{cuda_id}")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map=f"cuda:{cuda_id}")

model_name = "gemma"

input_dir = "../scenario_txt_Nov_MITRatio"
output_dir = "../answers_txt_MITRatio/Gemma"

#############################################################################################

for input_file_name in tqdm(sorted(os.listdir(input_dir))):
    input_text = read_txt(f"{input_dir}/{input_file_name}")

    input_ids = tokenizer(input_text, return_tensors="pt").to(f"cuda:{cuda_id}")
    outputs = model.generate(**input_ids)
    answer = tokenizer.decode(outputs[0])[len(input_text) + len("<bos>") + len(" ")]

    write_into_txt(answer, f"{output_dir}/{input_file_name}")
