## Equipping Language Models with Tool Use Capability for Tabular Data Analysis in Finance

<p>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.9+-1f425f.svg?color=blue">
    </a>
    <a href="https://github.com/adriantheuma/llama2-raven/blob/main/LICENCE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
    <a href="https://huggingface.co/adriantheuma/raven-lora" target="_blank">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20-Model-blue?color=blue&logoColor=white" />
    </a>
    <a href="https://huggingface.co/adriantheuma/raven-data" target="_blank">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20-Data-blue?color=blue&logoColor=white" />
    </a>
</p>

![teaser](raven-infeerence-pipeline.png)

## Overview

This repository is based on our publication *Equipping Language Models with Tool Use Capability for Tabular Data Analysis in Finance* ([PDF](https://aclanthology.org/2024.eacl-short.10.pdf)). It contains code to fine-tune the base [Llama 2 13B Chat](https://huggingface.co/meta-llama/Llama-2-13b) using LoRA with a curated table-and-text and sentiment analysis dataset in the financial domain. It also includes a demo gradio user interface to interact with the model. If you use this code or data in your work, please cite:

```
@inproceedings{theuma-shareghi-2024-equipping,
    title = "Equipping Language Models with Tool Use Capability for Tabular Data Analysis in Finance",
    author = "Theuma, Adrian  and
      Shareghi, Ehsan",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-short.10",
    pages = "90--103",
    abstract = "Large language models (LLMs) have exhibited an array of reasoning capabilities but face challenges like error propagation and hallucination, particularly in specialised areas like finance, where data is heterogeneous, and precision is paramount. We explore the potential of language model augmentation with external tools to mitigate these limitations and offload certain reasoning steps to external tools that are more suited for the task, instead of solely depending on the LLM{'}s inherent abilities. More concretely, using financial domain question answering datasets, we apply supervised finetuning on a LLAMA-2 13B CHAT model to act both as a task router and task solver. The task router dynamically directs a question to either be answered internally by the LLM or externally via the right tool from the tool set. Our tool-equipped SFT model, RAVEN, demonstrates an improvement of 35.2{\%} and 5.06{\%} over the base model and SFT-only baselines, respectively, and is highly competitive with strong GPT-3.5 results. To the best of our knowledge, our work is the first that investigates tool augmentation of language models for the finance domain.",
}
```

## Data & Templates

For data that was used to train the LoRA weights visit the [dataset](https://huggingface.co/datasets/adriantheuma/raven-data) repository. It is also possible to experiment with different prompt templates. Simply copy one of the existing templates in the templates/ folder and modify with your own prompts. More details can be found [here](/templates/README.md). 


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

Example:

```
python generate_interface.py \
    base_model meta-llama/Llama-2-13b-chat-hf \
    lora_weights adriantheuma/raven-lora \
    dataset_name adriantheuma/raven-data \
    dataset_split test \
    prompt_template = raven_prompt_template \
    use_peft True
```

See details with command `python generate_interface.py -h`

## Supervised Fine-tuning

Example:

This fine-tuning approach was performed on a single RTX 4090 24GB GPU

```
python finetune.py \
    --base_model meta-llama/Llama-2-13b-chat-hf \
    --dataset_name adriantheuma/raven-data \
    --prompt_template_name raven_prompt_template \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --num_epochs 5
```
See details with command `python finetune.py -h`


## References
Our Llama LoRA training code is based on [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)