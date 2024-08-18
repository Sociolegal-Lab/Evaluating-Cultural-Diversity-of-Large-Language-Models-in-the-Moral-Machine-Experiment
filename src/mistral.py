from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import os
from tqdm import tqdm


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

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map=f"cuda:{cuda_id}")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", device_map=f"cuda:{cuda_id}")

model_name = "mistral"

input_dir = "../scenario_txt_Nov_MITRatio"
output_dir = "../answers_txt_MITRatio/Mistral"

#############################################################################################

batch = 1

prompt_txt = sorted(os.listdir(input_dir))

group_prompt = []
for i in range(0, len(prompt_txt), batch):
    group_prompt.append(prompt_txt[i:i + batch])

group_prompt = tqdm(group_prompt)
for idx, gp in enumerate(group_prompt):
    messages = [
        [{"role": "user",
          "content": read_txt(f"{input_dir}/{input_file_name}")}]
        for input_file_name in gp
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.cuda()
    model.cuda()

    generated_ids = model.generate(model_inputs, max_new_tokens=30, do_sample=True)  # max_new_tokens is the length limit of answer
    decoded = tokenizer.batch_decode(generated_ids)

    for answer, input_file_name in zip(decoded, gp):
        write_into_txt(answer, f"{output_dir}/{input_file_name}")
