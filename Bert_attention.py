from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import pandas as pd
import time

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_name)
data_path = "./SST-2/"

def modi_data(obj_df):
    obj_df = obj_df[obj_df["label"] != 2]
    # obj_df["label"] = obj_df["label"].astype(string)
    obj_df.label = obj_df.label.replace(0, "neg")
    obj_df.label = obj_df.label.replace(1, "neg")
    obj_df.label = obj_df.label.replace(3, "pos")
    obj_df.label = obj_df.label.replace(4, "pos")
    obj_df = obj_df.fillna(0)
    # obj_df["label"] = obj_df["label"].astype(int)
    return obj_df

# 输入文本
test_df = pd.read_csv(
    data_path + "test.csv", header=None, sep="\t", names=["label", "text"]
)
test_df = modi_data(test_df)
test_sentence = test_df.text.tolist()
# text = "Hello, how are you?"

max_att = []
ave_att = []


for i in range(len(test_sentence)):
    output_df = pd.DataFrame()
    print(i)
    print(time.asctime())
    print(test_sentence[i])
    # 预处理输入文本
    inputs = tokenizer(test_sentence[i], return_tensors='pt')

    # 获取模型输出
    outputs = model(**inputs)

    # 提取注意力权重
    attentions = outputs.attentions

    # 初始化存储注意力分数的字典
    token_attention_scores = {i: [] for i in range(attentions[0].size(-1))}

    # 遍历每一层的注意力矩阵
    for layer_attention in attentions:
        # layer_attention 的形状为 (batch_size, num_heads, sequence_length, sequence_length)
        # 提取批次中的第一个样本
        layer_attention = layer_attention[0]

        # 遍历每一个注意力头
        for head_attention in layer_attention:
            # 遍历序列中的每一个单词
            for token_idx in range(head_attention.size(0)):
                # 提取当前单词的注意力分数
                token_attention = head_attention[token_idx].detach().numpy()
                token_attention_scores[token_idx].append(token_attention)

    # 计算每个单词在所有层和所有注意力头下的最大注意力分数和平均注意力分数
    max_attention_scores = {}
    mean_attention_scores = {}

    for token_idx, scores in token_attention_scores.items():
        scores = np.array(scores)
        max_attention_scores[token_idx] = np.max(scores, axis=(0, 1))
        mean_attention_scores[token_idx] = np.mean(scores, axis=(0, 1))

    # 计算所有token的平均注意力分数和最大注意力分数
    all_max_attention_scores = list(max_attention_scores.values())
    all_mean_attention_scores = list(mean_attention_scores.values())

    overall_max_attention_score = np.max(all_max_attention_scores)
    overall_mean_attention_score = np.mean(all_mean_attention_scores)

    # 输出注意力分数
    for token_idx in token_attention_scores.keys():
        print(f"Token {token_idx}:")
        print(f"  Max Attention Score: {max_attention_scores[token_idx]}")
        print(f"  Mean Attention Score: {mean_attention_scores[token_idx]}")

    # 输出所有token的平均注意力分数和最大注意力分数
    print(f"Overall Max Attention Score: {overall_max_attention_score}")
    print(f"Overall Mean Attention Score: {overall_mean_attention_score}")
    max_att.append(overall_max_attention_score)
    ave_att.append(overall_mean_attention_score)

    output_df["MAX_Attention"] = max_att
    output_df["AVE_Attention"] = ave_att
    output_df.to_csv("./Metrics/Bert_att_0618.csv")

