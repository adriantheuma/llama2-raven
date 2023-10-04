# from arepl_dump import dump
import re
import numpy as np
import pandas as pd
import json
from evaluate import load


# s = "Country | Share of children who report being bullied, 2015 <0x0A> Solomon Islands | 67 <0x0A> Lithuania | 52 <0x0A> Jordan | 41 <0x0A> Russia | 33 <0x0A> Honduras | 32 <0x0A> Finland | 28 <0x0A> Hungary | 24 <0x0A> Myanmar | 19 <0x0A> Tajikistan | 7"

# def split_row(row, splitter="|"):
#     return list(map(str.strip, row.split(splitter)))

# rows = s.split("<0x0A>")

# dict = {}
# dict["header"] = split_row(rows[0])
# dict["rows"] = [split_row(row) for row in rows[1:]]
# dict["types"] = []

# for i in range(len(dict["header"])):
#     c = [r[i] for r in dict["rows"]]
#     dict["types"].append("real" 
#                          if all(v.isnumeric() for v in c) 
#                          else "text"
#                         )

# j = json.dumps(dict)

bertscore = load("bertscore")
score = bertscore.compute(predictions=["The amount of goodwill reallocated from to the IOTG operating segment in 2018 was $480 million"], 
                          references=["The amount of goodwill reallocated from to the IOTG operating segment in 2018 was $480"], lang="en") 
print(score)