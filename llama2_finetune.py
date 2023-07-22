from dataclasses import dataclass, field
from typing import Optional, List
import fire


import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer

tqdm.pandas()
from utils.prompter import Prompter


def generate_prompt(data_point, prompt_template_name):
    prompter = Prompter(prompt_template_name)
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    return full_prompt



def train(
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf", 
    dataset_name: str = "unwilledset/raven-data",
    dataset_subset: str = "dataset-4",
    dataset_split: str = "train",
    dataset_text_field: str = "text",
    output_dir: str = "weights",
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 1.41e-5,
    cutoff_len: int = 512,
    val_set_size: int = 2000,
    gradient_accumulation_steps: int = 16,
    logging_steps: int = 1,
    seq_length: int = 512,

    # lora hyperparams
    use_peft: bool = True,
    peft_lora_r: int = 16,
    peft_lora_alpha: int = 16,
    peft_lora_dropout: float = 0.05,
    peft_lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,

    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {dataset_name}\n"
        f"dataset_subset: {dataset_subset}\n"
        f"dataset_split: {dataset_split}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {peft_lora_r}\n"
        f"lora_alpha: {peft_lora_alpha}\n"
        f"lora_dropout: {peft_lora_dropout}\n"
        f"lora_target_modules: {peft_lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"add_eos_token: {add_eos_token}\n"
        f"group_by_length: {group_by_length}\n"
        f"load_in_8bit: {load_in_8bit}\n"
        f"load_in_4bit: {load_in_4bit}\n"

        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        f"prompt template: {prompt_template_name}\n"
    )


    # Step 2: Load the dataset
    dataset = load_dataset(
        path=dataset_name, 
        name=dataset_subset,        
        split=dataset_split
    )
    train_data = dataset.shuffle().map(
        generate_prompt, 
        prompt_template_name
    )

    # Step 1: Load the model
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

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_auth_token=True,
    )



    

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
    )


    # Step 4: Define the LoraConfig
    if use_peft:
        peft_config = LoraConfig(
            r=peft_lora_r,
            lora_alpha=peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None


    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        dataset_text_field=dataset_text_field,
        peft_config=peft_config,
    )

    trainer.train()
    


if __name__ == "__main__":
    fire.Fire(train)