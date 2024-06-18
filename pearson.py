import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv("./Metrics/combine.csv")
x_ave = df["AVE_Attention"]
x_max = df["MAX_Attention"]

Input = ["VBCount", "NNCount", "JJCount", "RBCount", "ConjCount", "SentCount", "Length",
                      "Polysemy", "DependDist", "SentiScore", "SentiFlip", "ConstHeight", "TerminalRatio"]

for i in Input:
    y = df[i]
    correlation_ave, p_ave = pearsonr(x_ave,y)
    correlation_max, p_max = pearsonr(x_max,y)
    with open('pearson.txt', 'a') as file:
        file.write(i)
        file.write("\n")
        file.write("AVE")
        file.write(str(correlation_ave))
        file.write("\t")
        file.write(str(p_ave))
        file.write("\n")
        file.write("MAX")
        file.write(str(correlation_max))
        file.write("\t")
        file.write(str(p_max))
        file.write("\n")


file.close()