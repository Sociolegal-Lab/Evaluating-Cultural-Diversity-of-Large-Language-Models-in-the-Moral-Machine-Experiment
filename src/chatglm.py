from transformers import AutoTokenizer, AutoModel
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


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

input_dir = "../scenario_txt_Nov_MITRatio"
output_dir = "../answers_txt_MITRatio/ChatGLM"

input_file_name_list = sorted(os.listdir(input_dir))
group_prompt = [read_txt(f"{input_dir}/{input_file_name}") for input_file_name in input_file_name_list]
for input_file_name, prompt in zip(input_file_name_list, group_prompt):
    response, history = model.chat(tokenizer, prompt, history=[])
    write_into_txt(response, f"{output_dir}/{input_file_name}")
