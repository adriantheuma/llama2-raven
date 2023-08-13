import json
import os.path as osp
from typing import Union

import pandas as pd

from pandas import DataFrame
from pandasql import sqldf


class Prompter(object):
    __slots__ = ("_template", "_verbose", "_data", "_instruction", "_input")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self._data = None
        self._instruction = None

        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "raven_prompt_template"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self._template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def __table_to_pandas(self, table_as_dict: {}) -> DataFrame:
        """
        Converts a dict of header and rows in a table
        back to list
        
        Args:
            table_as_dict (list): The table in dict format

        Returns:
            list: Table as list with first row = header
        """
        header = table_as_dict["header"]
        table = {
            "columns": header,
            "data": table_as_dict["rows"]
        }

        types = table_as_dict["types"]        

        table_df = pd.read_json(
            json.dumps(table), 
            orient="split", 
            convert_dates=False,
            dtype=types
        )

        table_df.columns = table_df.columns.astype("object")
        table_df.columns = list(map(lambda x: "{}_".format(x) if str(x).isnumeric() else x, table_df.columns))
        header = table_df.columns

        for i in range(len(header)):
            if types[i] == "real" and str(table_df[header[i]].dtype) == 'object':
                table_df[header[i]] = table_df[header[i]].apply(lambda x: x.replace(",", ""))
                table_df[header[i]] = table_df[header[i]].astype("float64")
        
        return table_df

    def __clean_equation(self, eq: str) -> str:
        out = eq.replace(",", "")
        out = out.replace("%", "")
        return out
    
    def __clean_table(self) -> str:
        cleaned_table = self._data
        if cleaned_table.startswith("{'header"):
            cleaned_table = cleaned_table \
                .replace('"', "~") \
                .replace("'", '"') \
                .replace("~", "'")

        return cleaned_table
    

    def __get_sql(self, response: str) -> str:
        sql = None
        if len(response.split("### SQL:")) > 1:
            sql = response.split("### SQL:")[1].strip()
        return sql
    
    def __get_equation(self, response: str) -> str:
        equation = None
        if len(response.split("### Eval:")) > 1:
            equation = response.split("### Eval:")[1].split("### SQL:")[0].strip()
        
        return self.__clean_equation(equation)
    
    def __evaluate_query(self, response: str) -> (str, str):
        result = "None"
        sql = self.__get_sql(response)
        
        cleaned_table = self.__clean_table()
        try:
            j = json.loads(cleaned_table)
            data_table = self.__table_to_pandas(j)

            # data_table will be referenced in through the locals()
            # when running the sql script
            res = sqldf(sql, locals())

            if len(res) > 0:
                result = res.iloc[0][0] 
        except Exception as e:
            result = f"SQL is malformed: {sql}\nException:{e}"

        return (sql, result)



    def __evaluate_equation(self, response: str) -> (str, str):
        eq = self.__get_equation(response)
        try:
            result = round(eval(eq),2)
        except Exception as e:
            result = f"Equation is malformed: {eq}\nException:{e}"

        return (eq, result)


    def generate_training_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        data: Union[None, str] = None,
        derivation_eval: Union[None, str] = None,
        derivation_sql: Union[None, str] = None,
        response: Union[None, str] = None,
        template: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        self._instruction = instruction
        self._input = input
        self._data = data

        # there must be a better way to do this
        # but I have to move fast

        if input and data:
            prompt_id = "full_prompt" 
        elif data and not input:
            prompt_id = "data_prompt" 
        elif not data and  input:
            prompt_id = "input_prompt"       
        else:
            prompt_id = "default_prompt"       

        prompt_format = self._template[template][prompt_id]
        
        if template == "generic":
            if prompt_id == "input_prompt":
                prompt = prompt_format.format(instruction=instruction, input=input)
            else:
                prompt = prompt_format.format(instruction=instruction)
            prompt += response
        elif template == "script":
            if prompt_id == "full_prompt":
                prompt = prompt_format.format(instruction=instruction, input=input, data=data)
            elif prompt_id == "data_prompt":
                prompt = prompt_format.format(instruction=instruction, data=data)
            prompt += derivation_sql
        elif template == "arithmetic":
            if prompt_id == "full_prompt":
                prompt = prompt_format.format(instruction=instruction, input=input, data=data)
            elif prompt_id == "data_prompt":
                prompt = prompt_format.format(instruction=instruction, data=data)
            prompt += derivation_eval
        elif template == "table":
            if prompt_id == "full_prompt":
                prompt = prompt_format.format(instruction=instruction, input=input, data=data)
            elif prompt_id == "input_prompt":
                prompt = prompt_format.format(instruction=instruction, input=input)
            elif prompt_id == "data_prompt":
                prompt = prompt_format.format(instruction=instruction, data=data)
            prompt += response
        elif template == "template":
            if prompt_id == "full_prompt":
                prompt = prompt_format.format(instruction=instruction, input=input, data=data)
            elif prompt_id == "input_prompt":
                prompt = prompt_format.format(instruction=instruction, input=input)
            elif prompt_id == "data_prompt":
                prompt = prompt_format.format(instruction=instruction, data=data)
            else:
                prompt = prompt_format.format(instruction=instruction)
            prompt += response

        return prompt

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        data: Union[None, str] = None,
        derivation_eval: Union[None, str] = None,
        derivation_sql: Union[None, str] = None,
        response: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        self._instruction = instruction
        self._input = input
        self._data = data

        prompts = [f"### Instruction:\n{instruction}"]
        if input:
            prompts.append(f"### Input:\n{input}")
        if data:
            prompts.append(f"### Data:\n{data}")
        if response:
            prompts.append(f"### Response:\n{response}")
        if derivation_eval:
            prompts.append(f"### Eval:\n{derivation_eval}")
        if derivation_sql:
            prompts.append(f"### SQL:\n{derivation_sql}")

        prompt = "\n\n".join(prompts)

        return prompt

    def get_response(self, output: str) -> (str, str, str, str):
        
        if "### Response:" in output:
            response = output.split("### Response:")[1].replace("</s>", "").strip()
        else:
            return (output, None, None, None)
    
        eval = None
        result = None

        if "### SQL:" in response:
            eval, result = self.__evaluate_query(response)
        if "### Eval:" in response:
            eval, result = self.__evaluate_equation(response)
                
        response = response.split("### SQL:")[0].split("### Eval:")[0].strip()
        
        return (output, response, eval, result)
    
    def get_response_for_evaluation(self, output: str) -> (str, str):
        response_split = self.template["response_split"]
        derivation_split = self.template["derivation_split"]
                
        response = output.split(response_split)[1].split(derivation_split)[0].replace("</s>", "").strip()
        derivation = None
        if len(output.split(derivation_split)) > 1:
            derivation = output.split(derivation_split)[1].replace("</s>", "").strip()

        return {
            "derivation" : derivation,
            "response": response
        }
        
