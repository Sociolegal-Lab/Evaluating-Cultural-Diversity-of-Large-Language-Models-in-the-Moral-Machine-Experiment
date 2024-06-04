import os
import json
from tqdm import tqdm
import openai  # pip install openai==0.28


def get_completion_gpt4(prompt_str, model="gpt-4-1106-preview"):
    messages = [{"role": "user", "content": prompt_str}]
    result = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return result.choices[0].message["content"]


def if_not_exist_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


def get_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = json.load(file, strict=False)
    return content


def set_json(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        jsObj = json.dumps(content, ensure_ascii=False, default=str)
        file.write(jsObj)
        file.flush()


def read_txt(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    return content


def write_into_txt(text, file_path):
    f = open(file_path, 'w', encoding='utf-8')
    f.write(text)
    f.close()


openai.api_key = ""

input_dir = "../scenario_txt_MITRatio"

output_dir = "../answers_txt_MITRatio"
if_not_exist_make_dir(output_dir)
output_dir = "../answers_txt_MITRatio/GPT4"
if_not_exist_make_dir(output_dir)

while True:
    try:
        for file in tqdm(os.listdir(input_dir)):
            finished_file_list = get_json("finished_file_list_MITRatio.json")
            if file in finished_file_list:
                pass
            else:
                prompt = read_txt(f"{input_dir}/{file}")

                answer = get_completion_gpt4(prompt)
                write_into_txt(answer, f"{output_dir}/{file}")

                finished_file_list.append(file)
                set_json(finished_file_list, "finished_file_list_MITRatio.json")
        break
    except:
        pass
