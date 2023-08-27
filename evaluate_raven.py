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


from transformers import (
    GenerationConfig, 
    AutoModelForCausalLM, 
    AutoConfig, 
    AutoTokenizer,
    BitsAndBytesConfig, 
) 

from utils.prompter import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

bertscore = load("bertscore")

def main(
    load_8bit: bool = True,
    base_model: str = "meta-llama/Llama-2-13b-chat-hf",
    lora_weights: str = "unwilledset/raven-13b-chat-d8",
    inference_mode: bool = True,
    force_download: bool = False,
    prompt_template: str = "raven_prompt_template", 
    dataset_name: str = "unwilledset/raven-data",
    dataset_subset: str = "dataset-8",
    dataset_split: str = "test",
    download_mode: str = "reuse_cache_if_exists", # force_redownload, reuse_dataset_if_exists, reuse_cache_if_exists 
    device_map: str = "auto",
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    use_peft: bool = True,
    evaluation_dir: str = "evaluation",
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 10,
    num_beams: int = 2,
    max_new_tokens: int = 1024,
    report_every_steps: int = 5,
    add_eos_token: bool = True,
    max_length: int = 1024,
):
   

    if load_in_8bit and load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
        )
        # This means: fit the entire model on the GPU:0
        device_map = {"": 0}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    prompter = Prompter(prompt_template)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_auth_token=True,
    )   
    base_config = AutoConfig.from_pretrained(base_model).to_dict()

    lora_config = {}
    if use_peft:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map=device_map,
            force_download=force_download,
            inference_mode=inference_mode,
        )
        lora_config = model.peft_config["default"].to_dict()    
    
    
    generation_config_dict = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        # "output_scores": True,                
    }

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    model = torch.compile(model)

    # Load the dataset
    dataset = load_dataset(
        path=dataset_name, 
        name=dataset_subset,        
        split=dataset_split,
        download_mode=download_mode
    )

    # dataset = dataset.select(range(0,15))


    def tokenize(prompt):
        result = tokenizer(
            prompt, 
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
            return_length=True
        )

        return result

    # function to generate and tokenise the prompt    
    def generate_and_tokenize_prompt(data_point):
        
        # generate the prompt using the selected template
        prompt = prompter.generate_inference_prompt(
            instruction=data_point["instruction"],
            input=data_point["input"],
            data=data_point["data"],
            template=data_point["template"]
        )

        # tokenize and return    
        return tokenize(prompt)



    def save_evaluation_results(prediction_results, results_df, console_output: bool = False):
        results_df = pd.concat([results_df, pd.DataFrame(prediction_results)], ignore_index=True)
                
        save_location = str.format(
            f"{evaluation_dir}/{base_config['_name_or_path']}/{lora_weights}/"
        )

        dir_name = os.path.dirname(save_location)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        now = datetime.now()
        filename = f"results_{now.strftime('%Y%m%d_%H')}.csv"
        path = os.path.join(save_location, filename)

        # save the file
        results_df.to_csv(path, index=False, quotechar='"')  



    def evaluate(
        instruction: str,
        input: Union[None, str] = None,
        data: Union[None, str] = None,
        template: str = None,
        generation_config_dict: dict = {}
    ) -> dict:
                
        prompt = prompter.generate_inference_prompt(instruction, input, data, template)

        inputs = tokenizer(
            prompt, 
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
            return_length=True
        )
        length = inputs["length"].item()    
        if length < 1024:
            input_ids = inputs["input_ids"].to(device)
            generation_config = GenerationConfig(**generation_config_dict)       

            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)

            return prompter.get_response_for_evaluation(output=output, template=template)
        else:
            return {
                "result": "not evaluated"
            }

    prediction_results = []

    pbar = tqdm(desc="Evaluating test samples", total=dataset.num_rows)
    steps = 0

    results_df = pd.DataFrame(
        columns=["source", "template", "instruction", "input", "data", "gold", "gold_eval", "pred", "pred_eval"])


    #eval_data = (dataset.map(generate_and_tokenize_prompt))

    for sample in dataset:        
        instruction = sample["instruction"]
        input = sample["input"]
        data = sample["data"]
        template = sample["template"]
        source = sample["source"]
        gold_eval = sample["derivation_eval"] if sample["derivation_eval"] else sample["derivation_sql"]
        gold = sample["output"]

        prediction = evaluate(
            instruction=instruction, 
            input=input,
            data=data,
            template=template,
            generation_config_dict=generation_config_dict
        )

        if prediction["result"] != "not evaluated":
            pred = prediction["result"]
            pred_eval = prediction["eval"]

            # bert_score = bertscore.compute(predictions=[predicted], references=[gold], lang="en")
        
            prediction_results.append({
                "source": source,
                "template": template,
                "instruction": instruction,
                "input": input,
                "data": data,
                "gold": gold,
                "gold_eval": gold_eval,
                "pred": pred,
                "pred_eval": pred_eval
            })
            
        pbar.update()

        steps+=1
        if steps % report_every_steps == 0: 
            save_evaluation_results(prediction_results, results_df, console_output=True)

    save_evaluation_results(prediction_results, results_df, console_output=True)

if __name__ == "__main__":
    fire.Fire(main)