import os
import json
from tqdm import tqdm
from datetime import datetime

import fire
import torch
from peft import PeftModel
from datasets import load_dataset

from transformers import (
    GenerationConfig, 
    AutoModelForCausalLM, 
    AutoConfig, 
    AutoTokenizer,
    BitsAndBytesConfig, 
) 

from utils.prompter import Prompter


device = "cuda"

def main(
    load_8bit: bool = True,
    base_model: str = "meta-llama/Llama-2-7b-hf",
    lora_weights: str = "unwilledset/raven-model",
    force_download: bool = False,
    prompt_template: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
    dataset_name: str = "unwilledset/raven-data",
    dataset_subset: str = "dataset-4",
    dataset_split: str = "test",
    download_mode: str = "reuse_cache_if_exists", # force_redownload, reuse_dataset_if_exists, reuse_cache_if_exists 
    device_map: str = "auto",
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    use_peft: bool = False,
    evaluation_dir: str = "evaluation",
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 10,
    num_beams: int = 2,
    max_new_tokens: int = 512,
    report_every_steps: int = 100
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
        )
        lora_config = model.peft_config["default"].to_dict()    
    
    
    generation_config_dict = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "output_scores": True,                
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

    def save_evaluation_results(prediction_results, console_output: bool = False):
        results_dict = {}
        total_num_match = 0
        total_num_samples = 0

        for source in prediction_results:
            num_match = sum(res["exact_match"] for res in prediction_results[source])
            num_samples = len(prediction_results[source])
            results_dict.setdefault(source, {})
            results_dict[source] = {
                "num_samples": num_samples,
                "num_match": num_match,
                "accuracy": num_match / num_samples,
            }
            total_num_match += num_match
            total_num_samples += num_samples

        results_dict["overall"] = {
            "num_samples": total_num_samples,
            "num_match": total_num_match,
            "accuracy": total_num_match / total_num_samples,
        }

        eval_dict = {
            "base_model_config": base_config,
            "lora_config": lora_config,
            "evaluation_config": generation_config_dict,
            "results": results_dict,
            "predictions": prediction_results
        }

        if console_output:
            print(results_dict)
        else:
            save_location = str.format(
                f"{evaluation_dir}/{base_config['_name_or_path']}/{lora_config['peft_type']}/{lora_config['r']}/{dataset_subset}"
            )

            dir_name = os.path.dirname(save_location)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            now = datetime.now()
            filename = f"results_{now.strftime('%Y%m%d_%H%M%S')}.json"    
            path = os.path.join(save_location, filename)

            # save the fule
            with open(path, "w") as f:
                json.dump(eval_dict, f, indent=4)

    
    def evaluate(
        instruction,
        input=None,
        generation_config_dict: dict = {}
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        generation_config = GenerationConfig(**generation_config_dict)       

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
    
        return prompter.get_response(output)


    prediction_results = {}

    pbar = tqdm(desc="Evaluating test samples", total=dataset.num_rows)
    steps = 0

    for sample in dataset:        
        prediction = evaluate(
            instruction=sample["instruction"], 
            input=sample["input"],
            generation_config_dict=generation_config_dict
        )

        prediction = prediction.strip().replace("</s>", "")

        prediction_results.setdefault(sample["source"], [])

        prediction_results[sample["source"]].append({
                "instruction": sample["instruction"],
                "input": sample["input"],
                "prediction": prediction,
                "gold": sample["output"],
                "exact_match": 1 if prediction == sample["output"] else 0,                
            }
        )
        pbar.update()
        
        steps+=1
        if steps % report_every_steps == 0: 
            save_evaluation_results(prediction_results, console_output=True)

    save_evaluation_results(prediction_results, console_output=False)

if __name__ == "__main__":
    fire.Fire(main)