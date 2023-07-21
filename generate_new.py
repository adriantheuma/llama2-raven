import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
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

def main(
    load_8bit: bool = True,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "unwilledset/raven-model",
    # lora_weights: str = "tloen/alpaca-lora",
    lora_weights_version: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": device},
            force_download=True,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)


    demo = gr.Blocks(
        title="Raven", 
        analytics_enabled=True
    )
    
    callback = gr.CSVLogger()

    with demo:    
        gr.Markdown(
        """
            <p style="text-align:center; font-size:3em;">            
                📈 Raven 📈        
            </p>
        """
        )
        gr.Markdown(
            """
            <p style="text-align: center;">
                Raven is a 7B-parameter LLaMA model finetuned to follow instructions in the finance domain. <br/>
                It is trained on the <a href="https://github.com/tatsu-lab/stanford_alpaca">Stanford Alpaca</a> dataset and several other finance datasets. For more information, please visit <a href="https://github.com/adriantheuma/fin-expert">the project's website</a>.  
            </p>
            """
        )
        with gr.Row():   
            with gr.Column(scale=50):     
                with gr.Row(variant="panel"):                     
                    instruction = gr.components.Textbox(
                        lines=2,
                        label="Instruction",                
                        placeholder="How is a raven related to wealth?",
                    )
                with gr.Row(variant="panel"):     
                    input = gr.components.Textbox(
                        lines=2, 
                        label="Input", 
                        info="Provide further context such as a passage.",
                        placeholder="none"
                    )   
                with gr.Row(variant="panel"):    
                    temperature = gr.components.Slider(
                        minimum=0, maximum=1, value=0.1, label="Temperature", info="A higher temperature value typically makes the output more diverse and creative but might also increase its likelihood of straying from the context."
                    )
                    top_k = gr.components.Slider(
                        minimum=0, maximum=100, step=1, value=10, label="Top-k", info="Top-k tells the model to pick the next token from the top ‘k’ tokens in its list, sorted by probability."
                    )
                    top_p = gr.components.Slider(
                        minimum=0, maximum=1, value=0.75, label="Top-p", info="Top-p picks from the top-k tokens based on the sum of their probabilities."
                    )   
                with gr.Row(variant="panel"):
                    beams = gr.components.Slider(
                        minimum=1, maximum=4, step=1, value=2, label="Beams", info="The final output is the most probable sequence of tokens found among the k beams."
                    )
                    max_tokens = gr.components.Slider(
                        minimum=1, maximum=2000, step=1, value=256, label="Max tokens", info="Stopping criteria for the model"
                    )
                    stream = gr.components.Checkbox(
                        label="Stream output", value=True, info="Stream the output one token at a time."
                    )   
                with gr.Row():
                    inputs = [temperature, top_k, top_p, beams, max_tokens, stream]
                    clear_btn = gr.ClearButton(value="Clear")
                    submit_btn = gr.components.Button(value="Submit", variant="primary")
                    
            with gr.Column(scale=50, variant="panel"):
                output = gr.inputs.Textbox(
                    lines=10,
                    label="Output",
                )
                flag_btn = gr.Button("Flag this output", variant="stop")
        
        # list of inputs and outputs
        inputs = [instruction, input, temperature, top_k, top_p, beams, max_tokens, stream]
        outputs = [output]
        
        # clear button setup
        components_to_clear = [instruction, input, output]
        clear_btn.add(components=components_to_clear)

        # submit on click 
        submit_btn.click(fn=evaluate, inputs=inputs, outputs=outputs)

        # We can choose which components to flag -- in this case, we'll flag all of them
        callback.setup(inputs + outputs, "flagged_data_points")
        flag_btn.click(
            fn=lambda *args: callback.flag(args), 
            inputs=inputs + outputs, 
            outputs=None, 
            preprocess=False,
            show_progress="full"
        )
        

    demo.queue().launch(
        server_name="0.0.0.0", 
        auth=("user", "HJ49AJnXy36kKYTg")
    )

if __name__ == "__main__":
    fire.Fire(main)