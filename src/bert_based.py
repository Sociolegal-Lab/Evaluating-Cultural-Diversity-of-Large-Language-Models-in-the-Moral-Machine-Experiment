import sys
import torch
from transformers import AutoTokenizer, BertForMultipleChoice, RobertaForMultipleChoice, DistilBertForMultipleChoice, AlbertForMultipleChoice, AdamW
from torch.utils.data import DataLoader, TensorDataset
from api import *

# input_dir = "scenario_txt_Nov_Balanced59"
input_dir = "../scenario_txt_Nov_MITRatio"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prompt_list = [read_txt(f"{input_dir}/{prompt_file}") for prompt_file in tqdm(sorted(os.listdir(input_dir)))]
prompt_list = [prompt.rsplit('\n', 1)[0] for prompt in tqdm(prompt_list)]
prompt_list = [f"{prompt}\n\n[OPTION]" for prompt in tqdm(prompt_list)]
print(prompt_list[0])

# option1 = "I choose: A."
# option2 = "I choose: B."
# option_list = [option1, option2]

option_list = ["A", "B"]

saved_model_dir = "bert_based_saved_models_Nov"

for model_category in ["bert", "roberta", "distilbert", "albert"]:

    # output_dir = "../bert_based_answer_Nov"
    output_dir = "../bert_based_answer_Nov_MITRatio"
    if_not_exist_make_dir(output_dir)

    output_dir = f"{output_dir}/{model_category}"
    if_not_exist_make_dir(output_dir)

    save_dict = {"File": [], "Answer": []}

    if model_category == "bert":
        model_name = "bert-base-uncased"
        model = BertForMultipleChoice.from_pretrained(model_name)
        model.load_state_dict(torch.load(f"{saved_model_dir}/{model_name}.pth"))
    elif model_category == "roberta":
        model_name = "roberta-base"
        model = RobertaForMultipleChoice.from_pretrained(model_name)
        model.load_state_dict(torch.load(f"{saved_model_dir}/{model_name}.pth"))
    elif model_category == "distilbert":
        model_name = "distilbert-base-cased"
        model = DistilBertForMultipleChoice.from_pretrained(model_name)
        model.load_state_dict(torch.load(f"{saved_model_dir}/{model_name}.pth"))
    elif model_category == "albert":
        model_name = "albert-base-v2"
        model = AlbertForMultipleChoice.from_pretrained(model_name)
        model.load_state_dict(torch.load(f"{saved_model_dir}/{model_name}.pth"))
    else:
        sys.exit()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = []

    max_length = 0

    labels = []

    for prompt in prompt_list:
        prompt_input_ids = []
        for option in option_list:
            text = prompt.replace("[OPTION]", option)

            tokens = tokenizer.tokenize(text)
            if len(tokens) > max_length:
                max_length = len(tokens)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
            prompt_input_ids.append(indexed_tokens)
        input_ids.append(prompt_input_ids)

    new_input_ids = []
    for prompt_input_ids in input_ids:
        padded_input_ids = [seq + [0] * (max_length - len(seq)) for seq in prompt_input_ids]
        new_input_ids.append(padded_input_ids)

    input_ids = torch.tensor(new_input_ids)
    print(input_ids.size())

    attention_mask = (input_ids != 0).int()
    print(attention_mask.size())

    # Create a DataLoader for batching
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Use GPU
    model.to(device)

    model.eval()

    # Make predictions
    doc_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # Forward pass through the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
            )
            batch_logits = outputs.logits
            # print(batch_logits)

            # Get the predicted labels (choices)
            for logits in batch_logits:
                logits = torch.Tensor([logits.tolist()])
                # print("logits =", logits)
                predicted_labels = logits.argmax(dim=1)

                save_dict["File"].append(sorted(os.listdir(input_dir))[doc_idx])
                save_dict["Answer"].append(option_list[predicted_labels.item()])

                write_into_txt(option_list[predicted_labels.item()], f"{output_dir}/{model_category}_{sorted(os.listdir(input_dir))[doc_idx]}")
                doc_idx += 1

    save_df = pd.DataFrame.from_dict(save_dict)
    save_df.to_excel(f"{output_dir}/{model_category}.xlsx", index=False)
