
import json
import os
import py_trees.composites
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools.render import render_text_description
from langchain_core.tools import tool
from langchain_community.llms import Tongyi
from py_trees import common, behaviour
from langchain_core.prompts import PromptTemplate
import time

os.environ["DASHSCOPE_API_KEY"] = "sk-d012904b8d3e40d3991bb4b12c8c2f16"
model = Tongyi()


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
        final_prompt = """{system_prompt}\n{behaviour_prompt}\n{data_prompt}\n{output_format_prompt}"""
        prompt = PromptTemplate.from_template(final_prompt)
        parser = JsonOutputParser()
        chain = prompt | model | parser
        str = f'{self.default_system_prompt()}\n{self.behaviour_prompt}\n{self.data_prompt}\n{self.output_format_prompt}'
        print(str)
        res = chain.invoke({"system_prompt": self.system_prompt,
                            "behaviour_prompt": self.behaviour_prompt,
                            "data_prompt": self.data_prompt,
                            "output_format_prompt": self.output_format_prompt})
        return res

    @staticmethod
    def default_system_prompt():
        sp = '''Reward the trajectory of an autonomous vehicle. The input consists of 13 samples, '
              'each with 32 dimensions. The dimensions are defined as follows:)'
              'Dimensions 1-12: Represent the xy coordinates of the nearest lane center point of the ego vehicle and the xy coordinates of the next 5 lane center points.'
              'Dimensions 13-22: Represent the xy coordinates of the ego vehicle and the nearest 4 surrounding vehicles.'
              'Dimensions 23-32: Represent the xy velocities of the ego vehicle and the nearest 4 surrounding vehicles.'
              'In the coordinate system, the x-axis points to the right, and the y-axis points forward.'''

        return sp

    @staticmethod
    def default_output_format_prompt():
        op = '''The output format is a json array with 13 dimensions, 
                representing the rewards for adjacent pairs of samples.
                Only output your reward for each sample in json format
                Only output your reward for each sample in json format'''
        return op
