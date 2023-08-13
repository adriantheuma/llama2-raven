import os
import sys


import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import (
    GenerationConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig, 
) 

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
    base_model: str = "meta-llama/Llama-2-13b-chat-hf",
    lora_weights: str = "unwilledset/raven-13b-chat-d6",
    force_download: bool = False,
    lora_weights_version: str = "",
    prompt_template: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    device_map: str = "auto",
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    use_peft: bool = True,
    load_model: bool = True
):
    prompter = Prompter(prompt_template)
    
    if load_model:
        base_model = base_model or os.environ.get("BASE_MODEL", "")
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"


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

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            use_auth_token=True,
        )   

        if use_peft:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map=device_map,
                force_download=force_download,
            )
    
        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        model = torch.compile(model)

    def test_prompter(
        instruction,
        input=None,
        data=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction=instruction, input=input, data=data)
        test_output = [
            prompt,
            "### Response: this is the model's response",
             #"### SQL: SELECT ([Stadium]) FROM data_table WHERE LOWER([Score]) = LOWER('66-20')"
             "### Eval: 7>4"
        ]

        output = "\n\n".join(test_output)


        response, eval, result = prompter.get_response(output)
        yield (response, eval, result)


    def evaluate(
        instruction,
        input=None,
        data=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction=instruction, input=input, data=data)
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
        with gr.Row(equal_height=True):   
            with gr.Column(scale=15):    
                with gr.Row(variant="panel", ):      
                    gr.Markdown(
                    """
                        <p style="text-align:centre; font-size:3em;">            
                            📈 Raven 📈        
                        </p>
                    """
                    )
            with gr.Column(scale=85):    
                with gr.Row(variant="panel"):      
                    gr.Markdown(
                        """<br/>
                        <p style="text-align: middle;">
                            Raven is a <a href="https://ai.meta.com/llama/">LLaMA 2 model</a> finetuned to follow instructions in the finance domain. <br/>
                            It is trained on the <a href="https://github.com/tatsu-lab/stanford_alpaca">Stanford Alpaca</a> dataset and several <a href="https://huggingface.co/datasets/unwilledset/raven-data/viewer/dataset-3/train">other finance datasets</a>. For more information, please visit <a href="https://github.com/adriantheuma/fin-expert">the project's website</a>.  
                        </p>
                        """
                    )
        with gr.Row():   
            with gr.Column(scale=44):     
                with gr.Row(variant="panel"):                     
                    instruction = gr.components.Textbox(
                        lines=2,
                        label="Instruction",                
                        placeholder="What would you like to know?",
                        value="Test instruction with some data"
                    )
                with gr.Row(variant="panel"):     
                    input = gr.components.Textbox(
                        lines=4, 
                        label="Input", 
                        placeholder="Provide further context such as a narrative or a passage.",
                    )   
                with gr.Row(variant="panel"):     
                    data = gr.components.Textbox(
                        lines=6, 
                        label="Table", 
                        placeholder=
                            'Provide data related to the instruction. This needs to be in the following format to be interpreted correctly\n{\n\t"header": ["H1", "H2"],\n\t"rows": [["R1D1", "R1D2"], ["R2D1", "R2D2"]],\n\t"types: ["text", "text"]\n}',
                        value='{"header": ["Date", "Result", "Score", "Stadium", "City", "Crowd"], "rows": [["13 June 1997", "Adelaide Rams def. Leeds Rhinos", "34-8", "Adelaide Oval", "Adelaide", "14,630"], ["13 June 1997", "Hunter Mariners def. Castleford Tigers", "42-12", "Wheldon Road", "Castleford", "3,087"], ["14 June 1997", "North Queensland Cowboys def. Oldham Bears", "54-16", "Stockland Stadium", "Townsville", "12,631"], ["14 June 1997", "Auckland Warriors def. Bradford Bulls", "20-16", "Odsal Stadium", "Bradford", "13,133"], ["15 June 1997", "Canterbury Bulldogs def. Halifax Blue Sox", "58-6", "Belmore", "Sydney", "5,034"], ["15 June 1997", "Canberra Raiders def. London Broncos", "66-20", "Bruce Stadium", "Canberra", "6,471"], ["15 June 1997", "Sheffield Eagles def. Perth Reds", "26-22", "Don Valley Stadium", "Sheffield", "3,000"], ["15 June 1997", "Penrith Panthers def. Warrington Wolves", "52-22", "Wilderspool", "Warrington", "3,850"], ["16 June 1997", "Brisbane Broncos def. Wigan Warriors", "34-0", "ANZ Stadium", "Brisbane", "14,833"]], "types": ["text", "text", "text", "text", "text", "real"], "caption": "Round 2"}'
                    )   
                
                with gr.Row():
                    clear_btn = gr.ClearButton(value="Clear")
                    submit_btn = gr.components.Button(value="Submit", variant="primary")
                    test_btn = gr.components.Button(value="Test", variant="primary")
                    
            with gr.Column(scale=44):
                with gr.Row(variant="panel"):    
                    raw_out = gr.inputs.Textbox(
                        lines=2,
                        label="Raw output",
                    )
                with gr.Row(variant="panel"):    
                    response_out = gr.inputs.Textbox(
                        lines=2,
                        label="Response",
                    )
                with gr.Row(variant="panel"):    
                    eval_out = gr.inputs.Textbox(
                        lines=2,
                        label="Evaluation",
                    )
                with gr.Row(variant="panel"):    
                    result_out = gr.inputs.Textbox(
                        lines=2,
                        label="Result",
                    )
                with gr.Row():    
                    flag_btn = gr.Button("Flag this output", variant="stop")
                
            with gr.Column(scale=12):
                with gr.Row(variant="panel"):    
                    temperature = gr.components.Slider(
                        minimum=0, maximum=1, value=0.1, label="Temperature", info="Effects model diversity at the expense of straying from the context."
                    )
                with gr.Row(variant="panel"):    
                    top_k = gr.components.Slider(
                        minimum=0, maximum=100, step=1, value=10, label="Top-k", info="Pick the next token from the top ‘k’ tokens, sorted by probability."
                    )
                with gr.Row(variant="panel"):    
                    top_p = gr.components.Slider(
                        minimum=0, maximum=1, value=0.75, label="Top-p", info="Pick from the top-k tokens based on the sum of their probabilities."
                    )   
                with gr.Row(variant="panel"):
                    beams = gr.components.Slider(
                        minimum=1, maximum=4, step=1, value=2, label="Beams", info="Most probable sequence of tokens found among 'k' beams."
                    )
                with gr.Row(variant="panel"):    
                    max_tokens = gr.components.Slider(
                        minimum=1, maximum=2000, step=1, value=256, label="Max tokens", info="Stopping criteria for the model"
                    )
                with gr.Row(variant="panel"):    
                    stream = gr.components.Checkbox(
                        label="Stream output", value=True, info="Stream the output one token at a time."
                    )   
                
        
        # list of inputs and outputs
        inputs = [instruction, input, data, temperature, top_k, top_p, beams, max_tokens, stream]
        outputs = [raw_out, response_out, eval_out, result_out]
        
        # clear button setup
        components_to_clear = [instruction, input] + outputs
        clear_btn.add(components=components_to_clear)

        # submit on click 
        submit_btn.click(fn=evaluate, inputs=inputs, outputs=outputs)
        test_btn.click(fn=test_prompter, inputs=inputs, outputs=outputs)

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