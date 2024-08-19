# Evaluating Cultural Diversity of Large Language Models in the Moral Machine Experiment

## Abstract

This study evaluates the cultural alignment and diversity of prominent Large Language Models (LLMs) by assessing their ability to simulate human moral decision-making across various cultural contexts. The Moral Machine Experiment dataset, encompassing 130 countries and diverse demographic groups, serves as the evaluation platform. Our findings reveal a hierarchical structure in cultural diversity, with LLMs exhibiting greater adaptability at broader cultural classifications (e.g. cultural clusters) and progressively lower diversity at more granular levels (e.g. countries). Notably, ChatGLM-6B, demonstrates superior cultural diversity due to its diverse pre-training corpus. However, the analysis also identifies potential biases within LLMs, including the marginalization of minority groups and the perpetuation of gender inequality across cultural zones. The study emphasizes the critical need to prioritize cultural diversity throughout LLM development to foster inclusivity and mitigate potential risks associated with bias.

## Data

[The Moral Machine Experiment Database](https://osf.io/3hvt2/)

## Results

#### Dataset 1 (Unbalanced) - Accuracy

| Model | Cultural Cluster | Cultural Zone | Country | Language |
| ----- | ---------------- | ------------- | ------- | -------- |
| GPT-3.5 | 52.0 | 52.5 | 51.2 | 52.4 |
| GPT-4 | 52.0 | 52.3 | 51.3 | 52.6 |
| Llama-2-7B | 53.7 | 53.8 | 53.3 | 53.1 |
| Llama-3-8B | 58.0 | 57.7 | 57.2 | 57.2 |
| Aquila-7B | 50.0 | 50.1 | 49.8 | 49.2 |
| ChatGLM-6B | 59.0 | 59.3 | 59.3 | 60.0 |
| ChatGLM-3-6B | 54.3 | 54.7 | 53.8 | 55.1 |
| Mistral-7B | 59.0 | 58.6 | 58.0 | 58.2 |
| Gemma-7B | 50.7 | 50.7 | 50.3 | 51.4 |
| BERT | 50.3 | 50.4 | 48.5 | 49.7 |
| RoBERTa | 50.3 | 50.3 | 48.5 | 49.2 |
| DistilBERT | 50.3 | 50.4 | 49.5 | 50.2 |
| ALBERT | 50.0 | 50.0 | 48.7 | 49.2 |

#### Dataset 1 (Unbalanced) - Standard Deviation

| Model | Cultural Cluster | Cultural Zone | Country | Language |
| ----- | ---------------- | ------------- | ------- | -------- |
| GPT-3.5 | 0.0 | 0.8 | 7.3 | 4.3 |
| GPT-4 | 0.0 | 0.8 | 7.0 | 3.7 |
| Llama-2-7B | 0.6 | 1.0 | 7.5 | 3.2 |
| Llama-3-8B | 0.0 | 0.8 | 6.4 | 6.1 |
| Aquila-7B | 0.0 | 0.9 | 8.5 | 8.3 |
| ChatGLM-6B | 0.0 | 0.5 | 7.0 | 3.3 |
| ChatGLM-3-6B | 0.6 | 1.1 | 7.6 | 6.7 |
| Mistral-7B | 0.0 | 0.7 | 6.0 | 5.6 |
| Gemma-7B | 0.6 | 0.9 | 8.4 | 7.0 |
| BERT | 0.6 | 0.7 | 6.2 | 4.6 |
| RoBERTa | 0.6 | 0.5 | 6.9 | 3.7 |
| DistilBERT | 0.6 | 0.7 | 8.0 | 3.8 |
| ALBERT | 0.0 | 0.5 | 7.3 | 2.7 |

#### Dataset 2 (Balanced) - Accuracy

| Model | Cultural Cluster | Cultural Zone | Country | Language |
| ----- | ---------------- | ------------- | ------- | -------- |
| GPT-3.5 | 58.3 | 58.2 | 58.6 | 59.1 |
| GPT-4 | 64.7 | 64.1 | 64.7 | 64.1 |
| Llama-2-7B | 53.3 | 50.6 | 52.4 | 52.5 |
| Llama-3-8B | 50.3 | 49.9 | 49.9 | 49.4 |
| Aquila-7B | 50.0 | 50.1 | 50.2 | 50.7 |
| ChatGLM-6B | 51.0 | 51.2 | 50.2 | 49.6 |
| ChatGLM-3-6B | 57.3 | 56.6 | 57.5 | 57.4 |
| Mistral-7B | 49.3 | 49.8 | 49.6 | 49.4 |
| Gemma-7B | 54.7 | 55.2 | 54.9 | 56.4 |
| BERT | 50.3 | 50.1 | 50.6 | 50.5 |
| RoBERTa | 49.3 | 48.9 | 49.0 | 50.1 |
| DistilBERT | 50.0 | 50.4 | 50.2 | 50.9 |
| ALBERT | 50.3 | 50.1 | 50.5 | 49.1 |

#### Dataset 2 (Balanced) - Standard Deviation

| Model | Cultural Cluster | Cultural Zone | Country | Language |
| ----- | ---------------- | ------------- | ------- | -------- |
| GPT-3.5 | 0.6 | 2.7 | 6.0 | 4.6 |
| GPT-4 | 0.6 | 2.6 | 5.5 | 4.3 |
| Llama-2-7B | 3.5 | 7.8 | 14.0 | 13.4 |
| Llama-3-8B | 0.6 | 1.3 | 4.8 | 4.1 |
| Aquila-7B | 0.0 | 1.6 | 5.6 | 5.3 |
| ChatGLM-6B | 0.0 | 1.1 | 5.4 | 4.1 |
| ChatGLM-3-6B | 0.6 | 1.9 | 5.2 | 4.0 |
| Mistral-7B | 0.6 | 1.8 | 4.8 | 4.7 |
| Gemma-7B | 0.6 | 1.5 | 6.7 | 6.6 |
| BERT | 0.6 | 2.6 | 6.8 | 5.8 |
| RoBERTa | 0.6 | 2.3 | 6.5 | 5.8 |
| DistilBERT | 1.7 | 1.6 | 6.1 | 4.6 |
| ALBERT | 0.6 | 2.1 | 6.6 | 4.8 |
