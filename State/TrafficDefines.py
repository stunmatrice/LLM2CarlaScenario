
import carla
import numpy as np
from typing import List

class UnifiedState:
    state_dim: int
    state_lower_bound: np.ndarray
    state_upper_bound: np.ndarray
    state_description: List[str]
    def __init__(self, state_dim, state_lower_bound, state_upper_bound, state_description):
        self.state_dim = state_dim
        self.state_lower_bound = state_lower_bound
        self.state_upper_bound = state_upper_bound
        self.state_description = state_description

    def query_state_from_carla(self):
        pass

    @classmethod
    def defaultVehicleState(cls):
        dim = 10
        des = [''
               '']
        vs = UnifiedState(dim, )
        return vs





class UnifiedAction:
    action_dim: int
    action_lower_bound: np.ndarray
    action_upper_bound: np.ndarray
    action_description: List[str]

    def __init__(self, action_dim, action_lower_bound, action_upper_bound, action_description):
        self.action_dim = action_dim
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound
        self.action_description = action_description


