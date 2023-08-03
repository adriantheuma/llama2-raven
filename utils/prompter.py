"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        response: Union[None, str] = None,
        derivation: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.

        if input and derivation:
            template_name = "prompt_input_derivation" 
        elif input and not derivation:
            template_name = "prompt_input" 
        elif not input and derivation:
            template_name="prompt_no_input_derivation" 
        else: 
            template_name="prompt_no_input" 


        res = self.template[template_name].format(
                        instruction=instruction, 
                        input=input, 
                        response=response,
                        derivation=derivation,
        )

        # if input:
        #     if derivation:
        #         res = self.template["prompt_input_derivation"].format(
        #             instruction=instruction, 
        #             input=input, 
        #             response=response,
        #             derivation=derivation,
        #         )
        #     else:
        #         res = self.template["prompt_input"].format(
        #             instruction=instruction, 
        #             input=input,
        #             response=response,
        #             derivation=derivation,
        #         )
        # else:
        #     if derivation:
        #         res = self.template["prompt_no_input_derivation"].format(
        #             instruction=instruction,
        #             response=response,
        #             derivation=derivation,
        #         )
        #     else:
        #         res = self.template["prompt_no_input"].format(
        #             instruction=instruction,
        #             response=response,
        #     )

        # if label:
        #     res = f"{res}{label}"
        # if self._verbose:
        #     print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].replace("</s>", "").strip()

    def get_response_for_evaluation(self, output: str) -> str:
        response_split = self.template["response_split"]
        derivation_split = self.template["derivation_split"]
        out = output.split(response_split)[1].split(derivation_split)[0].replace("</s>", "").strip()
        return out
        
