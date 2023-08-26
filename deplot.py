from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image

import requests
import torch
import gradio as gr
import fire

# This means: fit the entire model on the GPU:0


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(
    load_8bit: bool = True,
    base_model: str = "google/deplot",

    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    device_map: dict = {"": 0},
):

    processor = Pix2StructProcessor.from_pretrained(
        base_model, 
        device_map=device_map
    )

    model = Pix2StructForConditionalGeneration.from_pretrained(
        base_model,
        device_map=device_map
    )

    # url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
    # image = Image.open(requests.get(url, stream=True).raw)

    def to_table(
        instruction,
        image: Image = None,
        **kwargs,
    ):
                
        inputs = processor( 
            images=image, 
            text=instruction, 
            return_tensors="pt"
        ).to(device)

        predictions = model.generate(**inputs, max_new_tokens=512)
        
        yield processor.decode(predictions[0], skip_special_tokens=True)
    
    
    
    demo = gr.Blocks(
        title="Image conversion test", 
        analytics_enabled=True
    )

    with demo:    
        with gr.Row(equal_height=True):   
            gr.Markdown(
            """
                <p style="text-align:centre; font-size:3em;">            
                    📈 Image conversion test 📈        
                </p>
            """
            )            
        with gr.Row():   
            with gr.Column(scale=50):     
                with gr.Row(variant="panel"):                     
                    instruction = gr.components.Textbox(
                        lines=2,
                        label="Instruction",                
                        value="Generate underlying data table of the figure below:" 
                    )
                with gr.Row(variant="panel"):                     
                    input = gr.components.Image(
                        label="Chart", 
                        source="upload", type="pil", container=True,
                        height=500
                    )   
                with gr.Row():
                    clear_btn = gr.ClearButton(value="Clear")
                    submit_btn = gr.components.Button(value="Submit", variant="primary")
            with gr.Column(scale=50):     
                with gr.Row(variant="panel"):                     
                    result_out = gr.components.Textbox(
                        lines=2,
                        label="Result",
                    )

        # list of inputs and outputs
        inputs = [instruction, input]
        outputs = [result_out]
        
        # clear button setup
        components_to_clear = inputs + outputs
        clear_btn.add(components=components_to_clear)

        # submit on click 
        submit_btn.click(fn=to_table, inputs=inputs, outputs=outputs)

    demo.queue().launch(
        server_name="0.0.0.0", share=True        
    )

if __name__ == "__main__":
    fire.Fire(main)