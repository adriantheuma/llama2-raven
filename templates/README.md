## Prompt templates

The template/ directory contains template styles for the prompts used to fine-tune LoRA models.

### Format

A template is described via a JSON file with the following keys: 

- `script`: Used for training observations that return an intermediate script
- `arithmetic`: Used for training observations that return an intermediate arithmetic formula
- `table`: Used for training observations where the expected response needs to be extracted from tabular data without any intermediate tool use.
- `generic`: Used in all other instances.
- `template`: Used to train the model to select the correct template from the above depending on the `instruction`, `input` and `data`.

Each of these sections is defined via the following set of keys:

- `description`: A short description of the template, with possible use cases.
- `full_prompt`: The template to use when input and data are not blank. Uses `{instruction}`,  `{input}` and `{data}` placeholders.
- `input_prompt`: The template to use when data is blank but input is available. Uses `{instruction}` and `{input}` placeholders.
- `data_prompt`: The template to use when input is blank but data is available. Uses `{instruction}` and `{data}` placeholders.
- `default-prompt`: The template to use when both input and data are blank. Uses `{instruction}`  placeholder.
- `response_split`: The text to use as separator when cutting real response or intermediate step (such an an arithmetic equation) from the model output.

### Current templates

#### raven_prompt_template.json

The template to use for the model trained on using tools or for fine-tuning using tools.

#### notools_prompt_template.json

The template to use for the model that was trained to produce the answer directly and not delegate to any tools or for fine-tuning without tools use.

#### llama_prompt_template.json

The template to use with the base Llama 2 model
