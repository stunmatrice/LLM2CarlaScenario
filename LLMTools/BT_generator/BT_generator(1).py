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


class dynamic_behavior(behaviour.Behaviour):
    def __init__(self, name, function_name, controller):
        super().__init__(name)
        self.function_name = function_name
        self.function_object = globals().get(self.function_name)
        self.controller = controller

    def update(self):
        self.function_object("")
        return common.Status.RUNNING


class dynamic_condition(behaviour.Behaviour):
    def __init__(self, name, function_name, controller):
        super().__init__(name)
        self.function_name = function_name
        self.function_object = globals().get(self.function_name)
        self.controller = controller

    def update(self):
        return self.function_object.invoke("")


class VehicleController(object):

    def __init__(self, user_input):
        self.user_input = user_input
        self.json_str = None
        self.json_dict = None
        self.behavior_tree = None

    def generate_bt_json_str(self):
        """Turn user input text into a bt in json format"""
        print('Generating bt in json format ..... ')
        info = self.user_input

        system_prompt = """You are an assistant helping to generate a behavior tree in json format. 
                        In the JSON for behavior trees, use the following words as keys: 
                        1.  The root node should use the key ROOT, 
                            The node type should use the keyword TYPE.
                            Child node should use the key CHILDREN.
                        2.  Node types include SEQUENCE, SELECTOR, PARALLEL, and BEHAVIOR.
                        3.  A SEQUENCE node will execute its child nodes in order until all child nodes return success, 
                            If a child node returns failure or running, the sequence node returns that child node's status.
                            You should use a sequence node to composite an event that requires 
                            condition checks, such as sequence('can overtake','overtake')
                            A SELECTOR node is a composite node that allows you to try multiple options until one succeeds. 
                            such as selector(sequence('can overtake','overtake'), behavior('keep in lane'))
                        4.  The behavior tree generated here is used to control vehicles in traffic scenarios, 
                            so please consider basic event priorities when generating it, such as emergency avoidance 
                            of pedestrians taking precedence over overtaking.
                        5.  For BEHAVIOR nodes, which are leaf nodes, use a FUNCTION_CALL field. 
                          This field should contain a DESCRIPTION field describing what the behavior is supposed to do, 
                          as well as a TYPE field. The TYPE field is divided into 0 and 1, 
                          where 0 represents a function call used for condition checking, and 1 represents a functional 
                          execution.
                        6. If a function call is used for condition checking, it may return success or failure.
                            For a functional execution it return running.
                            You should use sequence, selector, and parallel nodes to combine behavior nodes, 
                            ensuring that the user's intended meaning is correctly executed. 
                            
                        7. The result should be pure json, dont add any comment
                           The use input is : {info}  
                        """

        # ------------------------------------------------
        prompt = PromptTemplate.from_template(system_prompt)
        parser = JsonOutputParser()
        chain = prompt | model | parser
        res = chain.invoke({"info": info})
        self.json_dict = res

    def generate_bt_json_targeted(self):
        print('Adding functional call to behavior tree ..... ')
        json_string = json.dumps(self.json_dict)

        tools = [can_overtake, overtake, keep_driving_in_lane, should_avoid_pedestrians, avoid_pedestrians]
        tools_string = render_text_description(tools)
        system_prompt = """The following input is a JSON string representing a description of a behavior tree, 
                            where leaf nodes with FUNCTION_CALL indicate a function call, and the function's purpose 
                            is described by the DESCRIPTION field. Please add a FUNCTION_NAME field under FUNCTION_CALL 
                            with the value being the string name of the function found in the provided list of functions 
                            and their documentation. The input JSON string is as follows: {json_string}, 
                            and the list of functions along with their documentation is as follows: {tools_string}.
                            The result should be pure json, dont add any comment
                        """

        # ------------------------------------------------
        prompt = PromptTemplate.from_template(system_prompt)
        parser = JsonOutputParser()
        chain = prompt | model | parser
        res = chain.invoke({"json_string": json_string, "tools_string": tools_string})
        self.json_dict = res

    def assemble_behavior_tree(self, parent_node, children):
        if parent_node is not None:
            for child in children:
                child_type = child['TYPE']

                if child_type == 'SELECTOR':
                    children_l = child['CHILDREN']
                    parent_node_l = py_trees.composites.Selector('SELECTOR', memory=False)
                    parent_node.add_child(parent_node_l)
                    self.assemble_behavior_tree(parent_node_l, children_l)
                elif child_type == 'SEQUENCE':
                    children_l = child['CHILDREN']
                    parent_node_l = py_trees.composites.Sequence('SEQUENCE', memory=False)
                    parent_node.add_child(parent_node_l)
                    self.assemble_behavior_tree(parent_node_l, children_l)
                elif child_type == 'BEHAVIOR':
                    function_call = child['FUNCTION_CALL']
                    function_name = function_call['FUNCTION_NAME']
                    function_description = function_call['DESCRIPTION']
                    behaviour_type = function_call['TYPE']
                    if behaviour_type == 0:
                        behaviour_node = dynamic_condition(function_description, function_name, self)
                        parent_node.add_child(behaviour_node)
                    elif behaviour_type == 1:
                        behaviour_node = dynamic_behavior(function_description, function_name, self)
                        parent_node.add_child(behaviour_node)

        else:
            root_type = self.json_dict['ROOT']['TYPE']
            children = self.json_dict['ROOT']['CHILDREN']
            if root_type == 'SELECTOR':
                root_node = py_trees.composites.Selector('ROOT', memory=False)
            elif root_type == 'SEQUENCE':
                root_node = py_trees.composites.Sequence('ROOT', memory=False)
            else:
                raise ValueError(f'Unsupported root type: {root_type}')
            self.assemble_behavior_tree(root_node, children)
            self.behavior_tree = py_trees.trees.BehaviourTree(root_node)
            print(self.json_dict)

    def tick(self):
        self.behavior_tree.tick()


@tool()
def can_overtake(controller):
    """To use this method to determine whether the vehicle can overtake"""
    print('calling - can_overtake')
    return common.Status.SUCCESS

@tool()
def overtake(controller):
    """Use this method to control the vehicle's overtaking."""
    print('calling - overtake')
    return common.Status.RUNNING


@tool()
def keep_driving_in_lane(controller):
    """Use this method to keep the vehicle in its lane."""
    print('calling - keep_driving_in_lane')
    return common.Status.RUNNING


@tool()
def should_avoid_pedestrians(controller):
    """Use this method to determine whether there need to avoid pedestrians"""
    print('calling - should_avoid_pedestrians')
    return common.Status.FAILURE

@tool()
def avoid_pedestrians(controller):
    """Use this method to avoid pedestrians"""
    print('calling - avoid_pedestrians')
    return common.Status.RUNNING


input0 = ("The vehicle should normally stay in its lane. If it determines that overtaking is possible, \
                it should proceed with the overtake; if there are pedestrian, try to avoid pedestrians; \
          otherwise, it should continue driving in its current lane")
ins = VehicleController(input0)
ins.generate_bt_json_str()
ins.generate_bt_json_targeted()
print("Assembling behavior tree ..... ")
ins.assemble_behavior_tree(None, None)

while True:
    ins.tick()
    time.sleep(1)
