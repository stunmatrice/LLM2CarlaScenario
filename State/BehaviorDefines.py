
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
        b_prompt = '''The ego vehicle keeps driving alone the lane, 
                    If the x-coordinate of the ego vehicle remains at a fixed value, 
                    it is considered that lane keeping is good. 
                    Additionally, the ego vehicle should maintain a safe distance from the vehicle ahead. 
                    If both conditions are met, give a reward of 1; 
                    otherwise, give a reward of no more than 0.5.'''
        return VehicleBehaviorLaneKeeping(b_id, b_name, b_prompt)



