
from enum import Enum

class VehicleBehavior(Enum):
    BehaviorLaneKeeping = (1, "")

    def __init__(self, value, description):
        self.value = value
