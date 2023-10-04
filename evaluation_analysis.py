import pandas as pd
from pandas import DataFrame
from evaluate import load
import os

bertscore = load("bertscore")

#_results_path = "evaluation/meta-llama/Llama-2-13b-chat-hf/unwilledset/raven-13b-chat-d8/results_20230827_06.csv"

#_path = "evaluation/meta-llama/Llama-2-13b-chat-hf/no_peft/"
#_path = "evaluation/meta-llama/Llama-2-13b-chat-hf/unwilledset/raven-13b-chat-d8-no-tools/"
_path = "evaluation/meta-llama/Llama-2-13b-chat-hf/unwilledset/raven-13b-chat-d8/"
_analysed_results = "analysed_results.csv"
_results_path = os.path.join(_path, "results.csv")
_analysed_results_path = os.path.join(_path, _analysed_results)


def is_digit(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def model_results(sample):
    result = 0
    #if sample["source"] in ["wiki-sql", "tat-qa", "phrase-bank", "ott-qa"]:
    if sample["instruction"] == "What is the average value of packaging for years 2018 and 2019?":
        print(sample["instruction"]) 

    if is_digit(sample["gold"]) and is_digit(sample["pred"]):
        result = 1 if float(sample["gold"]) == float(sample["pred"]) else 0
    else:
        result = 1 if str(sample["gold"]).lower() == str(sample["pred"]).lower() else 0
    # else:
    #     result = bertscore.compute(predictions=sample["pred"], references=sample["gold"], lang="en")
    return result

if not os.path.exists(_analysed_results_path):
    df = pd.read_csv(_results_path)
    df.fillna("")

    print(df.info())


    df_template = df[df["template"]=="template"].sort_values(by="source")
    df_template["measure"] = df_template.apply(model_results, axis=1)

    df2 = df.drop(df_template.index)
    df_alpaca = df2[df2["source"]=="alpaca"]
    scores = bertscore.compute(predictions=df_alpaca["pred"].tolist(), references=df_alpaca["gold"].tolist(), lang="en")
    df_alpaca["measure"] = scores["f1"]
    
    df3 = df2.drop(df_alpaca.index)
    df3["measure"] = df3.apply(model_results, axis=1)

    df_final = pd.concat([df_alpaca, df3])
    print(df_final.shape)

    df_final.to_csv(os.path.join(_path, _analysed_results))

else:
    df_final = pd.read_csv(_analysed_results_path)

df_grouped = df_final.groupby(["source"])["measure"].agg(["sum", "count"])
df_grouped["%"] = df_grouped["sum"] / df_grouped["count"] * 100

print(df_grouped)

# df_final[(df_final["source"]=="tat-qa") & (df_final["measure"]==0)][["gold", "pred"]].to_csv(os.path.join(_path, "tat-qa-nomatch.csv"))
# df_final[(df_final["source"]=="ott-qa") & (df_final["measure"]==0)][["gold", "pred"]].to_csv(os.path.join(_path, "ott-qa-nomatch.csv"))
# df_final[(df_final["source"]=="wiki-sql") & (df_final["measure"]==0)][["gold", "pred"]].to_csv(os.path.join(_path, "wiki-sql-nomatch.csv"))






