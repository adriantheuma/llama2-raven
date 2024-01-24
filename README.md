# FireAct: Toward Language Agent Fine-tuning

<p>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.7+-1f425f.svg?color=blue">
    </a>
    <a href="https://github.com/anchen1011/FireAct/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
    <a href="https://huggingface.co/forestai/fireact_llama_2_7b" target="_blank">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20-Hugging%20Face-blue?color=blue&logoColor=white" />
    </a>
</p>

![teaser](raven-infeerence-pipeline.png)


This repository is based on our publication *FireAct: Toward Language Agent Fine-tuning* ([PDF](https://browse.arxiv.org/pdf/2310.05915.pdf)). It contains prompts, demo code and fine-tuning data we generated. It also includes the description and directory for the model family we fine-tuned. If you use this code or data in your work, please cite:

```
@misc{chen2023fireact,
      title={FireAct: Toward Language Agent Fine-tuning}, 
      author={Baian Chen and Chang Shu and Ehsan Shareghi and Nigel Collier and Karthik Narasimhan and Shunyu Yao},
      year={2023},
      eprint={2310.05915},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Overview



## Data & Templates


## Setup

Create virtual env, for example with conda

```
conda create -n llama2-raven python=3.9
conda activate llama2-raven
```

Clone this repo and install dependencies

```
git clone https://github.com/adriantheuma/llama2-raven.git
pip install -r requirements.txt
```

## Run Demo

#### Data Generation

Example:

```
python generate_interface.py 
```

See details with command `python generation.py -h`

You need to set a high number (thousands) of `--task_end_index` to get sufficient good data samples. **[WARNING] This is costly with gpt-4 and serpapi.**

You need to convert trajectories into [alpaca format](https://github.com/tatsu-lab/stanford_alpaca#data-release) or [gpt format](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) for training. See our examples [here](https://github.com/anchen1011/FireAct/tree/main/data/finetune).

#### Supervised Fine-tuning

Example:

```
cd finetune/llama_lora
python finetune.py \
    --base_model meta-llama/Llama-2-13b-chat-hf \
    --data_path ../../data/finetune/alpaca_format/hotpotqa.json \
    --micro_batch_size 8 \
    --num_epochs 30 \
    --output_dir ../../models/lora/fireact-llama-2-13b \
    --val_set_size 0.01 \
    --cutoff_len 512 \
```

See details [here](https://github.com/anchen1011/FireAct/tree/main/finetune).

#### Inference

Example (FireAct Llama):

```
python generation.py \
    --task hotpotqa \
    --backend llama \
    --evaluate \
    --random \
    --task_split dev \
    --task_end_index 5 \
    --modelpath meta-llama/Llama-2-7b-chat \
    --add_lora \
    --alpaca_format \
    --peftpath forestai/fireact_llama_2_7b_lora 
```

Example (FireAct GPT):

```
python generation.py \
    --task hotpotqa \
    --backend ft:gpt-3.5-turbo-0613:<YOUR_MODEL> \
    --evaluate \
    --random \
    --task_split dev \
    --temperature 0 \
    --chatgpt_format \
    --task_end_index 5
```

See details with command `python generation.py -h`

Set `--task_end_index 500` for quantitative evaluations. See our examples [here](https://github.com/anchen1011/FireAct/tree/main/trajs).

## Model Zoo

We release a selected set of multitask models based on Llama family. Details can be found in their model cards. 

| Base Model    | Training Method | Hugging Face                                               |
|---------------|-----------------|------------------------------------------------------------|
| Llama2-7B     | LoRA            | [forestai/fireact\_llama\_2\_7b\_lora](https://huggingface.co/forestai/fireact_llama_2_7b_lora)    |
| Llama2-13B    | LoRA            | [forestai/fireact\_llama\_2\_13b\_lora](https://huggingface.co/forestai/fireact_llama_2_13b_lora)   |
| CodeLlama-7B  | LoRA            | [forestai/fireact\_codellama\_7b\_lora](https://huggingface.co/forestai/fireact\_codellama\_7b\_lora)  |
| CodeLlama-13B | LoRA            | [forestai/fireact\_codellama\_13b\_lora](https://huggingface.co/forestai/fireact\_codellama\_13b\_lora) |
| CodeLlama-34B | LoRA            | [forestai/fireact\_codellama\_34b\_lora](https://huggingface.co/forestai/fireact\_codellama\_34b\_lora) |
| Llama2-7B     | Full Model      | [forestai/fireact\_llama\_2\_7b](https://huggingface.co/forestai/fireact_llama_2_7b)         |



## References
1. Our generation code is based on [ysymyth/ReAct](https://github.com/ysymyth/ReAct)
2. Our Llama full model training code is based on [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
3. Our Llama LoRA training code is based on [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
4. Our GPT fine-tuning code is based on [anchen1011/chatgpt-finetune-ui](https://github.com/anchen1011/chatgpt-finetune-ui/)