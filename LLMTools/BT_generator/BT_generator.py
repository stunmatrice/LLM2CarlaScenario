import json
import os

os.environ["DASHSCOPE_API_KEY"] = "sk-d012904b8d3e40d3991bb4b12c8c2f16"
from langchain_core.output_parsers import JsonOutputParser
# from langchain_openai import ChatOpenAI
# from langchain.tools.render import render_text_description
# from langchain_core.tools import tool

from langchain_community.llms import Tongyi
from langchain_core.outputs import ChatGeneration
from py_trees import composites, common, behaviour

model = Tongyi()


class dynamic_behavior(behaviour):
    def __init__(self, des):
        super().__init__()

    def update(self, des):
        return common.Status.RUNNING


class dynamic_condition(behaviour):
    def __init__(self):
        super().__init__()

    def update(self):
        return common.Status.SUCCESS


class BT_generator:

    def __init__(self, user_input, ):
        self.user_input = user_input
        self.json_str = None
        self.json_obj = None

    def generate_bt_json_str(self) -> json:
        '''Turn user input text into a bt in json format'''

        info = "The vehicle should normally stay in its lane. If it determines that overtaking is possible, \
                it should proceed with the overtake; otherwise, it should continue driving in its current lane"

        system_prompt = f"""You are an assistant helping to generate a behavior tree in json format. 
                        In the JSON for behavior trees, use the following words as keys: 
                        1.The root node should use the key ROOT.
                        2.The node type should use the keyword TYPE.
                        3.Types include SEQUENCE, SELECTOR, PARALLEL, and BEHAVIOR.
                        4.Child nodes should use the key CHILDREN.
                        5.For BEHAVIOR nodes, which are leaf nodes, use a FUNCTION_CALL field. 
                          This field should contain a DESCRIPTION field describing what the behavior is supposed to do, 
                          as well as a TYPE field. The TYPE field is divided into 0 and 1, 
                          where 0 represents a function call used for condition checking, and 1 represents a functional execution.
                        6.If a function call is used for condition checking, it may return success or failure.For a functional execution it return running.
                        7.The result should be pure json, dont add any comment
                          The use input is : {info}
                        """

        res = model.invoke(system_prompt)
        res = ChatGeneration(message=res)
        parser = JsonOutputParser()
        print(parser.parse_result([res]))

        return None

    def generate_bt_json_targeted(self):
        pass

    def assemble(self):
        pass


ins = BT_generator("dee")
print(ins.generate_bt_json_str())

# method_name = "call_using_str"
# method = getattr(ins, method_name)
# print(method)
# method()
