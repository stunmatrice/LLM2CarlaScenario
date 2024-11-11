
import json
import os
import py_trees.composites
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools.render import render_text_description
from langchain_core.tools import tool
from langchain_community.llms import Tongyi
from langchain.llms import OpenAI
from langchain.llms import AzureOpenAI
from py_trees import common, behaviour
from langchain_core.prompts import PromptTemplate
import time

os.environ["DASHSCOPE_API_KEY"] = "sk-d012904b8d3e40d3991bb4b12c8c2f16"
model = Tongyi()

# os.environ["OPENAI_API_VERSION"] = "gpt-4o-mini"
# os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://openscenariogeneration.openai.azure.com/'
# os.environ["AZURE_OPENAI_API_KEY"] = '07741c1fe26747fd811fc4816fa1c03c'
# model = AzureOpenAI()


class LLMAutoReward:
    system_prompt: str
    behaviour_prompt: str
    data_prompt: str
    output_format_prompt: str

    def __init__(self, system_prompt, behavior_prompt, data_prompt, output_format_prompt):
        self.system_prompt = system_prompt
        self.behaviour_prompt = behavior_prompt
        self.data_prompt = data_prompt
        self.output_format_prompt = output_format_prompt

    def get_result(self):
        final_prompt = """{system_prompt}\n{data_prompt}\n{behaviour_prompt}\n{output_format_prompt}"""
        prompt = PromptTemplate.from_template(final_prompt)
        parser = JsonOutputParser()
        chain = prompt | model | parser
        str = f'{self.default_system_prompt()}\n{self.data_prompt}\n{self.behaviour_prompt}\n{self.output_format_prompt}'
        print(str)
        res = chain.invoke({"system_prompt": self.system_prompt,
                            "data_prompt": self.data_prompt,
                            "behaviour_prompt": self.behaviour_prompt,
                            "output_format_prompt": self.output_format_prompt})
        return res

    @staticmethod
    def default_system_prompt():
        sp = '''An input array of size 13x13 will be provided, representing the state values of the ego vehicle observed 
                every 1.5 seconds. The state has 13 dimensions, defined as follows:
                Dimension 0: Values range from 1 to 2, with higher values indicating better lane keeping performance of 
                the vehicle.
                Dimensions 1-8: Represent the x and y coordinates of the ego vehicle and the three nearest surrounding 
                vehicles. A value of 0 represents that the corresponding vehicle is not present in the surroundings.
                Dimensions 9-12: Represent the speed values of the ego vehicle and the three surrounding vehicles.
                A value of 0 represents that the corresponding vehicle is not present in the surroundings.
                The array is as follows:'''

        return sp

    @staticmethod
    def default_output_format_prompt():
        op = '''The output format is a json array with 13 dimensions, 
                representing the rewards for adjacent pairs of samples.d
                Only output your reward for each sample in json format
                Only output your reward for each sample in json format
                Only output your reward for each sample in json format'''
        return op
