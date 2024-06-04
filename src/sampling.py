import os
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from collections import Counter
import math
import itertools
import json


def set_json(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        jsObj = json.dumps(content, ensure_ascii=False, default=str)
        file.write(jsObj)
        file.flush()


def if_not_exist_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


df_population = pd.read_csv("population.csv", encoding="utf-8")
df_scenario_sample = pd.read_excel("sample_num.xlsx")

output_dir = "CountryKmeansClusters"
if_not_exist_make_dir(output_dir)

for country, scenario_type, sample_num in zip(tqdm(df_scenario_sample["Country"]), df_scenario_sample["ScenarioType"], df_scenario_sample["Sample"]):
    if sample_num == 0:
        pass
    elif f"{country}_{scenario_type}.xlsx" in os.listdir(output_dir):
        pass
    else:
        df_country = df_population[df_population["country"].isin([country]) & df_population["scenario_type"].isin([scenario_type])]
        country_dict = df_country.to_dict("index")
        country_row_data = [list(country_dict[row_idx].values())[3:] for row_idx in country_dict]
        country_X = []
        for row_data in country_row_data:
            row_data_new = []

            # Age
            age_value = round((row_data[0] - 18) / (75 - 18), 2)
            row_data_new.append(age_value)

            # Gender
            if row_data[1] == "male":
                gender_value = [1, 0]
            elif row_data[1] == "female":
                gender_value = [0, 1]
            else:
                gender_value = [0, 0]
            row_data_new.extend(gender_value)

            # Education
            if row_data[2] == "college-educated":
                education_value = 1
            else:
                education_value = 0
            row_data_new.append(education_value)

            # Income
            if row_data[3] == "above national average income":
                income_value = 1
            else:
                income_value = 0
            row_data_new.append(income_value)

            # Political
            if row_data[4] == "progressive":
                political_value = 1
            else:
                political_value = 0
            row_data_new.append(political_value)

            # Religious
            if row_data[5] == "religious":
                religious_value = 1
            else:
                religious_value = 0
            row_data_new.append(religious_value)

            country_X.append(row_data_new)

        X = np.array(country_X)

        kmeans = KMeans(n_clusters=sample_num, random_state=0, n_init="auto").fit(X)
        cluster_label_list = kmeans.labels_

        df_country["label"] = cluster_label_list
        df_country.to_excel(f"{output_dir}/{country}_{scenario_type}.xlsx", index=False)

##############################################################################################

input_dir = "CountryKmeansClusters"

output_dir = "CountryResponseIDSampledExcel"
if_not_exist_make_dir(output_dir)


def get_diverse_score(df):
    diverse_score_ab = 0
    row_num = df.shape[0]
    row_idx_list = [i for i in range(row_num)]
    response_dict = df.to_dict("index")
    for row_idx_pair in itertools.combinations(row_idx_list, 2):
        response_a = response_dict[row_idx_pair[0]]
        response_b = response_dict[row_idx_pair[-1]]
        for variable in ["age", "gender", "education_anno", "income_anno", "political", "religious"]:
            if response_a[variable] != response_b[variable]:
                diverse_score_ab += 1
    return diverse_score_ab


for file in tqdm(os.listdir(input_dir)):
    if file in ["Maldives_Social Status.xlsx", "Oman_Social Status.xlsx"]:  # 0 samples
        pass
    else:
        country = file.split("_", 1)[0]
        scenario_type = file.split("_", 1)[-1].replace(".xlsx", "")
        sample_num = df_scenario_sample[df_scenario_sample["Country"].isin([country]) & df_scenario_sample["ScenarioType"].isin([scenario_type])].to_dict("records")[0]["Sample"]

        df_country = pd.read_excel(f"{input_dir}/{file}")
        label_list = sorted(df_country["label"])
        label_count_dict = dict(Counter(label_list).items())

        sorted_label_count_dict = dict(sorted(label_count_dict.items(), key=lambda x: x[1]))

        avg_sample_num = math.floor(sample_num / len(label_count_dict))
        if avg_sample_num < 1:
            avg_sample_num = 1

        label_sample_count = {}
        sample_already = 0
        for label_idx, label in enumerate(sorted_label_count_dict):
            if avg_sample_num > label_count_dict[label]:
                label_sample_count[label] = label_count_dict[label]
                sample_already += label_count_dict[label]

                avg_sample_num = math.floor((sample_num - avg_sample_num) / (len(label_count_dict) - label_idx - 1))
            else:
                label_sample_count[label] = avg_sample_num
                sample_already += avg_sample_num

        while True:
            if sum(list(label_sample_count.values())) < sample_num:
                new_label_sample_count = label_sample_count
                for label in label_sample_count:
                    if label_sample_count[label] < label_count_dict[label]:
                        new_label_sample_count[label] += 1
                        break
                    else:
                        pass
                label_sample_count = new_label_sample_count
            else:
                break

        # Calculate max iteration number
        max_iteration_num = 0
        for label in label_sample_count:
            iteration_num = math.ceil(sorted_label_count_dict[label] / label_sample_count[label])
            if iteration_num > max_iteration_num:
                max_iteration_num = iteration_num

        # Sampling
        max_diverse_score = 0
        final_df_total_sampled = pd.DataFrame()
        for iteration in range(max_iteration_num):
            df_total_sampled = pd.DataFrame()
            for label in label_sample_count:
                label_sample_num = label_sample_count[label]
                df_label = df_country[df_country["label"].isin([label])]
                df_label_sampled = df_label.sample(label_sample_num)
                df_total_sampled = pd.concat([df_total_sampled, df_label_sampled], ignore_index=True)

            diverse_score = get_diverse_score(df_total_sampled)

            if diverse_score >= max_diverse_score:
                max_diverse_score = diverse_score
                final_df_total_sampled = df_total_sampled
            else:
                pass

        final_df_total_sampled.rename(columns={"ïresponse_id": "response_id"}, inplace=True)
        final_df_total_sampled.to_excel(f"{output_dir}/{country}_{scenario_type}.xlsx", index=False)

##############################################################################################

input_dir = "CountryResponseIDSampledExcel"

country_sample_dict = {}

for file in tqdm(os.listdir(input_dir)):
    df_sample = pd.read_excel(f"{input_dir}/{file}")
    response_id_list = list(df_sample["response_id"]).copy()
    country_sample_dict[file[:-5]] = response_id_list

set_json(country_sample_dict, "country_sample_dict.json")
