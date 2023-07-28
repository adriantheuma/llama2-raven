import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("logs/logs.csv")
# df.rename(columns={"Unnamed: 0": "step"}, inplace=True)
print(df.head())


sns.set_theme(style="whitegrid")

loss = df[["step", "loss"]]

print(loss.head())

sns.lineplot(x=loss["step"], y=loss["loss"], palette="tab10", linewidth=2.5)
plt.show()


eval_loss = df.dropna(subset=["eval_loss"])[["step", "eval_loss"]]
print(eval_loss.head())

sns.lineplot(x=eval_loss["step"], y=eval_loss["eval_loss"], palette="tab10", linewidth=2.5)
plt.show()

# syeyqx
# zfjywc
# hlmmah
# lqqtfq
# qomdpl
# fjtjtu
# bufsed
# eoedhj
# zkfjqt
# lhiynv

