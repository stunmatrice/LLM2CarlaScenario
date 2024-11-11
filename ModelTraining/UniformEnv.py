
import gymnasium as gym
from PythonAPI.LLM2CarlaScenario.Wrappers.wrappers import *
from PythonAPI.LLM2CarlaScenario.State.TrafficDefines import VehicleActionSpace,VehicleStateSpace


class UniformVehicleEnv(gym.Env):
    vehicle: Vehicle
    client: carla.Client
    world: carla.World
    vehicle_state: VehicleStateSpace
    vehicle_action: VehicleActionSpace
    action_smoothing: float

    def __init__(self, vehicle, client, vehicle_state, vehicle_action, action_smoothing):
        self.vehicle = vehicle
        self.client = client
        self.vehicle_state = VehicleStateSpace.defaultVehicleState() if vehicle_state is None else vehicle_state
        self.vehicle_action = VehicleActionSpace.defaultVehicleAction() if vehicle_action is None else vehicle_action
        self.action_smoothing = action_smoothing
        self.action_space = gym.spaces.Box(low=self.vehicle_action.space_lower_bound,
                                           high=self.vehicle_action.space_upper_bound,
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.vehicle_state.space_lower_bound,
                                                high=self.vehicle_state.space_upper_bound,
                                                dtype=np.float32)

    def step(self, action):
        if action is not None:
            u1, d1 = action[0], action[1]
            u2, d2 = action[2], action[3]
            u3, d3 = action[4], action[5]

            steer = np.random.normal(loc=u1, scale=np.sqrt(d1))
            throttle = np.random.normal(loc=u2, scale=np.sqrt(d2))
            brake = np.random.normal(loc=u3, scale=np.sqrt(d2))
            # print(steer, throttle, brake)
            self.vehicle.control.steer = steer
            self.vehicle.control.throttle = throttle
            # self.vehicle.control.brake = brake
            self.vehicle.tick()
            self.vehicle.world.tick()
            new_state = self.vehicle.get_state_v3()

            done = self.vehicle.should_done
        return new_state, 0, done, False, {}

    def reset(self, seed= None, options = None):
        print('reset')
        self.vehicle.reset_position()
        return self.vehicle.get_state_v3(), {}













