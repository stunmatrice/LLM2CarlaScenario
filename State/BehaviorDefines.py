
from enum import Enum

class VehicleBehaviorBase:
    BehaviorID: int
    BehaviorName: str
    BehaviorPrompt: str

class VehicleBehaviorLaneKeeping(VehicleBehaviorBase):
    def __init__(self, b_id, b_name, b_prompt):
        self.BehaviorID = b_id
        self.BehaviorName = b_name
        self.BehaviorPrompt = b_prompt

    @classmethod
    def default_behavior(cls):
        b_id = 1
        b_name = 'LaneKeeping'
        b_prompt = '''The ego vehicle keeps driving alone the lane, give a reward between 0.0 - 1.0 
                    Prefer to assign a very low value, such as 0.0-0.1, 
                    unless 1. The x-coordinates of all samples are close to a certain number, 
                    2. The speed values are close to 60, 
                    3. There is no other vehicle within 5 meters. 
                    A reward of 1.0 should be given if all conditions are met. 
                    If any condition is not met, there should be a discount'''

        b_prompt_v2 = '''The ego vehicle keeps driving alone the lane, 
                        Consider the value a from the 0th dimension and the value b from the 9th dimension, where 
                        represents the ego vehicle's speed ranging from 0 to 60. Use the sigmoid function to map the 
                        speed value b to the range 0.0 to 2.0, resulting in c. 
                        The reward r1 is calculated as a×c. You should carefully get the number and compute ,and user r1 
                        as reward'''
        return VehicleBehaviorLaneKeeping(b_id, b_name, b_prompt_v2)



