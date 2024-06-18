import tensorflow as tf
from transformers import pipeline, BertTokenizer, TFBertForSequenceClassification
import os
import pandas as pd
import numpy as np
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertModel, BertConfig
from tensorflow.keras.models import load_model, Model
from scipy import stats
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Flatten,
    Conv1D,
    MaxPooling1D,
    Add,
    Lambda,
    Dropout,
    concatenate,
)
import keras

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
# 程序最多只能占用指定gpu50%的显存
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True  # 程序按需申请内存
sess = tf.compat.v1.Session(config=config)

data_path = "./SST-2/"
max_len = 49
remove_threshold = 1e-5


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


def val_to_res(val):
    if val > 0.5:
        return 1
    else:
        return 0


def deep_gini(val):
    ret = float(1 - pow(val, 2) - pow(1 - val, 2))
    return ret


def get_last_layer_model(model):
    layer_names = [layer.name for layer in model.layers]
    layer_output = model.get_layer(layer_names[-2]).output
    ret = Model(model.input, layer_output)

    return ret


def get_train_at(train_df, last_layer_model):
    train_df = modi_data(train_df)
    pos_df = train_df[train_df["label"] == "pos"]
    neg_df = train_df[train_df["label"] == "neg"]

    pos_sentence = pos_df.text.tolist()
    pos_x = tokenizer(pos_sentence, padding=True, truncation=True, return_tensors="tf")
    # pos_x = tokenizer.encode_plus(
    #     pos_sentence,
    #     add_special_tokens=True,  # 添加特殊标记[CLS]和[SEP]
    #     max_length=50,  # 设定最大长度
    #     padding='max_length',  # 对文本进行填充
    #     return_attention_mask=True,  # 返回attention mask
    #     return_tensors='tf'  # 返回TensorFlow张量
    # )
    pos_input_ids = pos_x['input_ids']
    pos_attention_mask = pos_x['attention_mask']
    # pos_x = tokenizer(pos_sentence, return_tensors="tf", padding=True, truncation=True)
    # pos_x = pad_sequences(pos_x, maxlen=max_len, padding="post")

    neg_sentence = neg_df.text.tolist()
    neg_x = tokenizer(neg_sentence, padding=True, truncation=True, return_tensors="tf")
    # neg_x = tokenizer.encode_plus(
    #     neg_sentence,
    #     add_special_tokens=True,  # 添加特殊标记[CLS]和[SEP]
    #     max_length=50,  # 设定最大长度
    #     padding='max_length',  # 对文本进行填充
    #     return_attention_mask=True,  # 返回attention mask
    #     return_tensors='tf'  # 返回TensorFlow张量
    # )
    neg_input_ids = pos_x['input_ids']
    neg_attention_mask = pos_x['attention_mask']
    # neg_x = tokenizer(pos_sentence, return_tensors="tf", padding=True, truncation=True)
    # neg_x = pad_sequences(neg_x, maxlen=max_len, padding="post")

    ret = {}
    ret["pos"] = last_layer_model.predict([pos_input_ids, pos_attention_mask])
    ret["neg"] = last_layer_model.predict([neg_input_ids, neg_attention_mask])
    return ret


def get_kernels(train_at):
    removed_cols = {'pos': [], 'neg': []}

    for i in range(train_at["pos"].T.shape[0]):
        if np.var(train_at["pos"].T[i]) < remove_threshold:
            removed_cols['pos'].append(i)
    for i in range(train_at["neg"].T.shape[0]):
        if np.var(train_at["neg"].T[i]) < remove_threshold:
            removed_cols['neg'].append(i)

    pos_vals = np.delete(train_at["pos"].T, removed_cols['pos'], axis=0)
    neg_vals = np.delete(train_at["neg"].T, removed_cols['neg'], axis=0)

    kernels = {}
    kernels["pos"] = stats.gaussian_kde(pos_vals)
    kernels["neg"] = stats.gaussian_kde(neg_vals)

    return kernels, removed_cols


def get_lsa(kernels, removed_cols, test_pred, test_label):
    lsa = []

    for i in range(len(test_pred)):
        value = np.delete(test_pred[i], removed_cols[test_label[i]])
        temp = np.negative(np.log(kernels[test_label[i]](value)))

        lsa.append(temp[0])

    return lsa


def find_closest_at(at, train_at):
    """The closest distance between subject AT and training ATs.
    Args:
        at (list): List of activation traces of an input.
        train_at (list): List of activation traces in training set (filtered)

    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.
    """

    dist = np.linalg.norm(at - train_at, axis=1)
    return (min(dist), train_at[np.argmin(dist)])


def get_dsa(test_pred, test_label, train_at):
    ret = []

    for i in range(len(test_pred)):
        label = test_label[i]
        at = test_pred[i]
        a_dist, a_dot = find_closest_at(at, train_at[label])
        b_dist, _ = find_closest_at(
            a_dot, train_at[list(set(["pos", "neg"]) - set([label]))[0]]
        )
        ret.append(a_dist / b_dist)
    return ret


# 加载预训练的BERT模型和tokenizer
model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)
model.summary()

# 构建自定义的输入层
input_ids = Input(shape=(None,), dtype='int32', name="input_ids")
attention_mask = Input(shape=(None,), dtype='int32', name="attention_mask")
inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

# 将BERT模型应用于自定义输入
bert_output = model(inputs)

# 添加自定义的输出层用于情感分类
output = Dense(1, activation='sigmoid', name='output')(bert_output.pooler_output)

# 构建模型
model = Model(inputs=[input_ids, attention_mask], outputs=output)

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# for layer in model.layers:
#     layer.trainable = True
#

# 准备输入数据
# sentence = "This is an example sentence to classify."
# inputs = tokenizer(sentence, return_tensors="tf")
train_df = pd.read_csv(
    data_path + "train.csv", header=None, sep="\t", names=["label", "text"]
)
test_df = pd.read_csv(
    data_path + "test.csv", header=None, sep="\t", names=["label", "text"]
)
test_df = modi_data(test_df)
test_sentence = test_df.text.tolist()

test_label = test_df.label.tolist()
test_y = np.array(test_label)
test_y[test_y == 'pos'] = 1
test_y[test_y == 'neg'] = 0
test_y = test_y.reshape(-1, 1).astype(int)

print(len(test_sentence))  # 1821

label = []
pred_val = []
LSA = []
DSA = []
DeepGini = []
is_Right = []

# 运行BERT模型进行文本分类
# outputs = model(inputs)
# print(outputs)
for i in range(len(test_sentence)):
    print(i)
    print(time.asctime())
    print(test_sentence[i])
    output_df = pd.DataFrame()
    # text1 = "The weather is really bad today."
    # inputs1 = tokenizer(test_sentence[i], return_tensors="tf", padding=True, truncation=True)
    encoded_text = tokenizer(test_sentence[i], return_tensors='tf', padding=True, truncation=True)
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']
    pred = model.predict([input_ids, attention_mask])
    pred_res = val_to_res(pred)
    # outputs = model(inputs1)
    # print(outputs1)
    # result = tf.nn.softmax(outputs.logits, axis=-1).numpy()

    label.append(val_to_res(pred))
    pred_val.append(pred)
    # result = result.reshape(2).tolist()
    # if result[0] > result[1]:
    #     label.append(0)
    #     pred = result[1]
    #     pred_val.append(pred)
    # else:
    #     label.append(1)
    #     pred = result[1]
    #     pred_val.append(pred)

    test_pred = pred
    train_at = get_train_at(train_df, model)
    kernels, removed_cols = get_kernels(train_at)
    lsa = get_lsa(kernels, removed_cols, test_pred, test_label)
    dsa = get_dsa(test_pred, test_label, train_at)
    LSA.append(lsa)
    DSA.append(dsa)
    deepgini = deep_gini(pred)
    DeepGini.append(deepgini)
    if pred_res == test_y[i]:
        flag = 1
    else: flag = 0
    is_Right.append(flag)

    # 保存数据

    output_df["PredRes"] = label
    output_df["PredVal"] = pred_val
    output_df["LSA"] = LSA
    output_df["DSA"] = DSA
    output_df["DeepGini"] = DeepGini
    output_df["is_Right"] = is_Right

    output_df.to_csv("./Metrics/Bert_outputs_0424.csv")

