import os
import json

with open("evaluation/evaluation_results.json") as file:
    eval = json.load(file)

with open("evaluation/evaluation_results.json", "w") as f:
    json.dump(eval, f, indent=4)