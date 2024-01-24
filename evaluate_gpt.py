import os
import json
from tqdm import tqdm
from datetime import datetime
from typing import Union

import re
import fire
import torch
from peft import PeftModel
from datasets import load_dataset
from evaluate import load
import pandas as pd
# Import the openai package
import openai
from openai.error import RateLimitError, InvalidRequestError, Timeout
import random as rnd
import tiktoken

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)  # for exponential backoff


from utils.prompter import Prompter



def main(
    openai_api_key: str = "",
    use_tools: bool = False,
    prompt_template: str = "cot_prompt_template", # cot_prompt_template, raven_prompt_template, notools_prompt_template
    dataset_name: str = "adriantheuma/raven-data",
    download_mode: str = "reuse_cache_if_exists", # force_redownload, reuse_dataset_if_exists, reuse_cache_if_exists 
    evaluation_dir: str = "evaluation",
    evaluation_subdir: str = "cot", # zero-shot-cot, tools, notools
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 10,
    num_beams: int = 2,
    report_every_steps: int = 5,
    few_shot: int = 0, # 0 invokes zero-shot-COT 
    cot: bool = True,
    model="gpt-3.5-turbo",
    max_tokens=4096
):
    # Set openai.api_key to the OPENAI environment variable
    openai.api_key = openai_api_key

    prompter = Prompter(prompt_template)
    encoding = tiktoken.encoding_for_model(model)
    
    generation_config_dict = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_beams": num_beams,
        "return_dict_in_generate": True,             
    }


    # Load the dataset
    dataset = load_dataset(
        path=dataset_name, 
        download_mode=download_mode
    )

    test_dataset = dataset["test"]
    train_dataset = dataset["train"]

    # function to generate prompt    
    def generate_prompt(data_point):
        # generate the prompt using the selected template
        prompt = prompter.generate_inference_prompt(
            instruction=data_point["instruction"],
            input=data_point["input"],
            data=data_point["data"],
            template=data_point["template"]
        )

        # tokenize and return    
        return prompt


    def save_evaluation_results(prediction_results, results_df, console_output: bool = False):
        results_df = pd.concat([results_df, pd.DataFrame(prediction_results)], ignore_index=True)

        save_location = str.format(
            f"{evaluation_dir}/openai/{evaluation_subdir}/"
        )

        dir_name = os.path.dirname(save_location)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        now = datetime.now()
        filename = f"results_{now.strftime('%Y%m%d_%H')}.csv"
        path = os.path.join(save_location, filename)

        # save the file
        results_df.to_csv(path, index=False, quotechar='"')  

    @retry(
            wait=wait_random_exponential(multiplier=1, max=60), 
            stop=stop_after_attempt(100), 
            retry=(retry_if_exception_type(RateLimitError) | retry_if_exception_type(Timeout))
    )
    def completion_with_backoff(**kwargs):
        output = ""
        
        try:
            output = openai.ChatCompletion.create(**kwargs)
        except InvalidRequestError:
            output = "Not evaluated"

        return output
    
    prediction_results = []

    results_df = pd.DataFrame(
        columns=["source", "template", "instruction", "input", "data", "gold", "gold_eval", "pred", "pred_eval"])


    filtered_test_dataset = test_dataset.filter(lambda sample: sample["template"] != "template" and sample["source"] != "alpaca")

    #filtered_test_dataset = filtered_test_dataset.select(list(range(100)))

    pbar = tqdm(desc="Evaluating test samples", total=filtered_test_dataset.num_rows)
    steps = 0

    def generate_training_prompt(sample, use_tools: bool):
        prompt = prompter.generate_training_prompt(
                instruction=sample["instruction"], 
                input=sample["input"], 
                data=sample["data"], 
                template="generic-3" if cot else sample["template"], 
                derivation_eval=sample["derivation_eval"], 
                derivation_sql=sample["derivation_sql"],
                response=sample["output"],
                use_tools=use_tools                 
            )
        return prompt
    

    def generate_inference_prompt(sample):
        prompt = prompter.generate_inference_prompt(
            instruction=sample["instruction"], 
            input=sample["input"], 
            data=sample["data"], 
            template="generic-3" if cot else sample["template"]
        )
        return prompt
    
    def get_shots(source: str, template: str):
        
        filtered = train_dataset.filter(lambda where: 
                             where["template"] == template and 
                             where["source"] == source)

        random_sample =  rnd.sample(range(0, filtered.num_rows), few_shot)
        return filtered.select(random_sample)


    def get_gpt_response(user_msgs):
        system_message = prompter.system_message_prompt_format(
            template="generic-3", 
            prompt_id="system_message_cot" if cot else "system_message_default"
        )
        
        system_msg = [
            {"role": "system", 
            "content": system_message
        }]
        msgs = system_msg+user_msgs
        to_encode = ""
        for msg in msgs:
            to_encode = to_encode + msg["content"] + "\n"
        encoded = encoding.encode(to_encode)
        if len(encoded) < max_tokens:
            response = completion_with_backoff(
                            model=model,
                            messages=msgs)
        else:
            response = "Not evaluated"
        
        return response["choices"][0]["message"]["content"] if response != "Not evaluated" else response

    evaluate = False
    encoded_lengths = []

    save_location = str.format(
            f"{evaluation_dir}/openai/{evaluation_subdir}/"
        )
    path = os.path.join(save_location, "results.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["key"] = df["source"] + df["template"] + df["instruction"] + df["input"] + df["data"]
        keys = list(df["key"])
    else:   
        keys = []


    for test_sample in filtered_test_dataset:
        key = test_sample["source"] + test_sample["template"] + test_sample["instruction"] + test_sample["input"] + test_sample["data"]
        shots = []

        if key not in keys:
            if few_shot != 0:
                few_shot_dataset = get_shots(test_sample["source"], test_sample["template"])            
                i=1
                for few_shot_sample in few_shot_dataset:
                    if cot:
                        shot = generate_inference_prompt(few_shot_sample)
                        user_msgs = [{"role": "user", "content": shot}]
                        output = get_gpt_response(user_msgs)

                        # replace the answer with the gold 
                        shot = shot + output.split("###Answer")[0] + "###Answer: " + few_shot_sample["output"]                                    
                    else:                
                        shot = generate_training_prompt(few_shot_sample, use_tools)            
                    
                    shots.append(f"Example {i}\n\n{shot}\n\n")
                    i+=1
            
                shots.append("Now answer the following:\n\n")

            
            test_shot = generate_inference_prompt(test_sample)
            shots.append(test_shot)
            
            # print("".join(shots))

            user_msgs = [{"role": "user", "content": "".join(shots)}]
            
            # evaluate
            response = get_gpt_response(user_msgs=user_msgs)
            
            if response != "Not evaluated":
                eval, result = prompter.get_response_for_gpt_evaluation(response, test_sample["template"], use_tools, cot)

                prediction_results.append({
                    "source": test_sample["source"],
                    "template": test_sample["template"],
                    "instruction": test_sample["instruction"],
                    "input": test_sample["input"],
                    "data": test_sample["data"],
                    "gold": test_sample["output"],
                    "gold_eval": test_sample["derivation_eval"] if test_sample["derivation_eval"] else test_sample["derivation_sql"],
                    "pred": result,
                    "pred_eval": eval
                })

            steps+=1
            if steps % report_every_steps == 0: 
                save_evaluation_results(prediction_results, results_df, console_output=True)
            
        pbar.update()


    save_evaluation_results(prediction_results, results_df, console_output=True)
    
    print(f"total number of tokens: {sum(encoded_lengths)}")


if __name__ == "__main__":
    fire.Fire(main)