import numpy as np
import pandas as pd
from scipy import stats

in_path = "./Metrics/Bert_Yelp_combine.csv"
df = pd.read_csv(in_path)
Input_metrics = ["VBCount", "NNCount", "JJCount", "RBCount", "ConjCount", "SentCount", "Length",
                      "Polysemy", "DependDist", "SentiScore", "SentiFlip", "ConstHeight", "TerminalRatio"]
Output_metrics = ["DSA", "LSA", "AL", "DeepGini"]
Attention_metrics = ["AVE_Attention", "MAX_Attention"]


for i in Attention_metrics:
    att_df = df[i]
    group1 = np.array(att_df)
    print(group1)
    for j in Input_metrics:
        in_df = df[j]
        group2 = np.array(in_df)
        # 计算两组数据的均值
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        # 使用scipy.stats.ttest_ind执行Wald检验
        t_stat, p_value = stats.ttest_ind(group1, group2)
        print("T-statistic: ", t_stat)
        print("P-value: ", p_value)
        # P-value小于某个显著性水平（例如0.05），我们拒绝原假设，接受Wald检验结果
        alpha = 0.05
        if p_value < alpha:
            print("Reject null hypothesis: Mean of group 1 and group 2 are different")
        else:
            print("Do not reject null hypothesis: Mean of group 1 and group 2 are probably the same")
    for k in Output_metrics:
        out_df = df[k]
        group2 = np.array(in_df)
        # 计算两组数据的均值
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        # 使用scipy.stats.ttest_ind执行Wald检验
        t_stat, p_value = stats.ttest_ind(group1, group2)
        print("T-statistic: ", t_stat)
        print("P-value: ", p_value)
        # P-value小于某个显著性水平（例如0.05），我们拒绝原假设，接受Wald检验结果
        alpha = 0.05
        if p_value < alpha:
            print("Reject null hypothesis: Mean of group 1 and group 2 are different")
        else:
            print("Do not reject null hypothesis: Mean of group 1 and group 2 are probably the same")
