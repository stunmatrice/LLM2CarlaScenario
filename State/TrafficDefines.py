
import numpy as np
from typing import List


class UnifiedSpaceDefine:
    space_dim: int
    space_lower_bound: np.ndarray
    space_upper_bound: np.ndarray
    space_description: List[str]

    # def __init__(self, space_dim, space_lower_bound, space_upper_bound, space_description):
    #     self.space_dim = space_dim
    #     self.space_lower_bound = space_lower_bound
    #     self.space_upper_bound = space_upper_bound
    #     self.space_description = space_description

    def parseFormArray(self, des_array):
        dim = len(des_array)
        lower_bound = [row[0] for row in des_array]
        lower_bound_np = np.array(lower_bound, dtype=np.float32)
        upper_bound = [row[1] for row in des_array]
        upper_bound_np = np.array(upper_bound, dtype=np.float32)
        des_strs = [row[2] for row in des_array]
        self.space_dim = dim
        self.space_lower_bound = lower_bound_np
        self.space_upper_bound = upper_bound_np
        self.space_description = des_strs
        return self


class VehicleStateSpace(UnifiedSpaceDefine):
    @classmethod
    def defaultVehicleState(cls):
        des = [[-10, 60, 'Ego vehicle velocity x'],
               [-20, 20, 'Ego vehicle velocity y'],
               [-1, 1, 'Ego vehicle velocity dot product with way point 1'],
               [-1, 1, 'Ego vehicle velocity dot product with way point 2'],
               [-1, 1, 'Ego vehicle velocity dot product with way point 3'],
               [-1, 1, 'Ego vehicle velocity dot product with way point 4'],
               [-50, 10, 'Distance x from Npc 1, if x is negative, Ego is behind the npc, '
                         'if x is positive, Ego is in front of the npc'],
               [-20, 20, 'Distance y from Npc 1, if y is negative, Ego is in the left of the npc, '
                         'if x is positive, Ego is in right of the npc'],
               [-50, 10, 'Distance x from Npc 2, if x is negative, Ego is behind the npc, '
                         'if x is positive, Ego is in front of the npc'],
               [-20, 20, 'Distance y from Npc 2, if y is negative, Ego is in the left of the npc, '
                         'if x is positive, Ego is in right of the npc'],
               [-50, 10, 'Distance x from Npc 3, if x is negative, Ego is behind the npc, '
                         'if x is positive, Ego is in front of the npc'],
               [-20, 20, 'Distance y from Npc 3, if y is negative, Ego is in the left of the npc, '
                         'if x is positive, Ego is in right of the npc'],
               [-50, 10, 'Distance x from Npc 4, if x is negative, Ego is behind the npc, '
                         'if x is positive, Ego is in front of the npc'],
               [-20, 20, 'Distance y from Npc 4, if y is negative, Ego is in the left of the npc, '
                         'if x is positive, Ego is in right of the npc'],
               [-10, 60, 'Npc 1 velocity x'],
               [-20, 20, 'Npc 1 velocity y'],
               [-10, 60, 'Npc 2 velocity x'],
               [-20, 20, 'Npc 2 velocity y'],
               [-10, 60, 'Npc 3 velocity x'],
               [-20, 20, 'Npc 3 velocity y'],
               [-10, 60, 'Npc 4 velocity x'],
               [-20, 20, 'Npc 4 velocity y']]

        des_v2 = [
            # Ego's point and next 5 points
            [-50, 50, 'Ego vehicle waypoint\'s x coordinate'],
            [-50, 50, 'Ego vehicle waypoint\'s y coordinate'],
            [-50, 50, 'Next waypoint 1\'s x coordinate'],
            [-50, 50, 'Next waypoint 1\'s y coordinate'],
            [-50, 50, 'Next waypoint 2\'s x coordinate'],
            [-50, 50, 'Next waypoint 2\'s y coordinate'],
            [-50, 50, 'Next waypoint 3\'s x coordinate'],
            [-50, 50, 'Next waypoint 3\'s y coordinate'],
            [-50, 50, 'Next waypoint 4\'s x coordinate'],
            [-50, 50, 'Next waypoint 4\'s y coordinate'],
            [-50, 50, 'Next waypoint 5\'s x coordinate'],
            [-50, 50, 'Next waypoint 5\'s y coordinate'],
            # 4 Npc 's positions
            [-50, 50, 'Ego vehicle\'s x coordinate'],
            [-50, 50, 'Ego vehicle\'s y coordinate'],
            [-50, 10, 'Npc 1 \'s x coordinate'],
            [-20, 20, 'Npc 1 \'s y coordinate'],
            [-50, 10, 'Npc 2 \'s x coordinate'],
            [-20, 20, 'Npc 2 \'s y coordinate'],
            [-50, 10, 'Npc 3 \'s x coordinate'],
            [-20, 20, 'Npc 3 \'s y coordinate'],
            [-50, 10, 'Npc 4 \'s x coordinate'],
            [-20, 20, 'Npc 4 \'s y coordinate'],
            # Ego's velocity and 4 npc's velocity
            [-10, 60, 'Ego vehicle velocity x'],
            [-20, 20, 'Ego vehicle velocity y'],
            [-10, 60, 'Npc 1 velocity x'],
            [-20, 20, 'Npc 1 velocity y'],
            [-10, 60, 'Npc 2 velocity x'],
            [-20, 20, 'Npc 2 velocity y'],
            [-10, 60, 'Npc 3 velocity x'],
            [-20, 20, 'Npc 3 velocity y'],
            [-10, 60, 'Npc 4 velocity x'],
            [-20, 20, 'Npc 4 velocity y']]

        vehicle_state_space = VehicleStateSpace()
        vehicle_state_space.parseFormArray(des_v2)
        return vehicle_state_space


class VehicleActionSpace(UnifiedSpaceDefine):
    @classmethod
    def defaultVehicleAction(cls):
        des = [[-1, 1, 'steer u'],
               [0,  1, 'steer d'],
               [0,  1, 'throttle u'],
               [0,  1, 'throttle d'],
               [0,  1, 'brake u'],
               [0,  1, 'brake d']]
        vehicle_action_space = VehicleActionSpace()
        vehicle_action_space.parseFormArray(des)
        return vehicle_action_space
