from typing import Optional, List
import fire
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    DataCollatorForSeq2Seq, 
    Trainer
)
import evaluate
import pandas as pd

# from trl import SFTTrainer
# from tqdm import tqdm
# from dataclasses import dataclass, field

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

from utils.prompter import Prompter

def train(
    # data params
    base_model: str = "meta-llama/Llama-2-13b-chat-hf",
    dataset_name: str = "unwilledset/raven-data",
    dataset_subset: str = "dataset-7",
    dataset_split: str = "train",
    download_mode: str = "reuse_cache_if_exists", # force_redownload, reuse_dataset_if_exists, reuse_cache_if_exists 
    output_dir: str = "weights",
    logging_dir: str = "logs",
    prompt_template_name: str = "raven_prompt_template",  # The prompt template to use, will default to alpaca.
    
    # training/ model hyperparams
    # batch size = per_device_batch_size * gradient_accumulation_steps
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 32, 
    
    num_train_epochs: int = 5,
    learning_rate: float = 3e-4,
    val_set_size: int = 0.1, # 10%
    logging_steps: int = 1,
    optim: str = "adamw_torch",
    eval_steps: int = 100, 
    save_steps: int = 100,
    warmup_steps: int = 100,
    fp16: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,

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
    
    # tokenizer settiongs
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    max_length: int = 512,
        
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Training model with params:\n\n"
        f"Data params:\n"
        f"   base_model: {base_model}\n"
        f"   data_path: {dataset_name}\n"
        f"   dataset_subset: {dataset_subset}\n"
        f"   dataset_split: {dataset_split}\n"
        f"   resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        f"   prompt template: {prompt_template_name}\n\n"
        
        f"Training hyperparams:\n"
        f"   per_device_train_batch_size: {per_device_train_batch_size}\n"
        f"   per_device_eval_batch_size: {per_device_eval_batch_size}\n"
        f"   gradient_accumulation_steps: {gradient_accumulation_steps}\n"
        f"   num_epochs: {num_train_epochs}\n"
        f"   learning_rate: {learning_rate}\n"
        f"   val_set_size: {val_set_size}\n"
        f"   group_by_length: {group_by_length}\n"
        f"   optim: {optim}\n"
        f"   eval_steps: {eval_steps}\n"
        f"   save_steps: {save_steps}\n"
        f"   fp16: {fp16}\n"
        f"   warmup_steps: {warmup_steps}\n"
        f"   load_in_8bit: {load_in_8bit}\n"
        f"   load_in_4bit: {load_in_4bit}\n\n"
        f"   output_dir: {output_dir}\n"
        f"   logging_dir: {logging_dir}\n\n"
        
        f"Tokenizer settings:\n"
        f"   max_length: {max_length}\n"
        f"   train_on_inputs: {train_on_inputs}\n"
        f"   add_eos_token: {add_eos_token}\n\n"

        f"PEFT hyperparams:\n"        
        f"   lora_r: {peft_lora_r}\n"
        f"   lora_alpha: {peft_lora_alpha}\n"
        f"   lora_dropout: {peft_lora_dropout}\n"
        f"   lora_target_modules: {peft_lora_target_modules}\n\n"
             

    )

    # initialise the prompter
    prompter = Prompter(prompt_template_name)

    # function to tokenise the prompt
    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        # labels and input ids are the same
        result["labels"] = result["input_ids"].copy()

        return result

    # function to generate and tokenise the prompt    
    def generate_and_tokenize_prompt(data_point):
        
        # generate the prompt using the selected template
        full_prompt = prompter.generate_training_prompt(
            instruction=data_point["instruction"],
            input=data_point["input"],
            data=data_point["data"],
            response=data_point["output"],
            derivation_eval=data_point["derivation_eval"],
            derivation_sql=data_point["derivation_sql"],
            template=data_point["template"],
        )

        # tokenize and return    
        return tokenize(full_prompt)
   
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
    model = prepare_model_for_kbit_training(model)

    # define the lora config
    if use_peft:
        peft_config = LoraConfig(
            r=peft_lora_r,
            lora_alpha=peft_lora_alpha,
            lora_dropout=peft_lora_dropout,
            target_modules=peft_lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None
    
    # wrap the model in a peft model
    model = get_peft_model(model, peft_config)

    
    # Load the dataset
    dataset = load_dataset(
        path=dataset_name, 
        name=dataset_subset,        
        split=dataset_split,
        download_mode=download_mode
    )
    
    model.print_trainable_parameters()  
    
    # get the tokeniser from the model
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    tokenizer.add_eos_token = add_eos_token

    if val_set_size > 0:        
        if val_set_size < 1:
            val_set_size = int(dataset.num_rows * val_set_size)

        # split the dataset into training and validation
        train_val = dataset.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )

        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # lengths = []
    # for data in train_data:
    #     lengths.append(len(data["input_ids"]))

    # plt.hist(lengths, bins=50)
    # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    # plt.show()

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        logging_steps=logging_steps,
        optim=optim,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if val_set_size > 0 else None,
        save_steps=save_steps,
        output_dir=output_dir,
        logging_dir=logging_dir,
        save_total_limit=5,
        load_best_model_at_end=True if val_set_size > 0 else False,
        group_by_length=group_by_length
    )


    trainer = Trainer(
        # model
        model=model,
        
        # dataset
        train_dataset=train_data,
        eval_dataset=val_data,
        
        # args
        args=training_args,
        
        # evaluation metric
        # compute_metrics=compute_metrics,

        # data collator
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    
    model = torch.compile(model)

    # start training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # save the model
    model.save_pretrained(output_dir)

    # log the results
    log_history = pd.DataFrame(trainer.state.log_history)
    log_history.to_csv(os.path.join(logging_dir, "logs.csv"))


if __name__ == "__main__":
    fire.Fire(train)