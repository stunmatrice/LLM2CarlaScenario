
import gymnasium as gym
from PythonAPI.LLM2CarlaScenario.Wrappers.wrappers import *
from PythonAPI.LLM2CarlaScenario.State.TrafficDefines import *

class UniformVehicleEnv(gym.Env):
    vehicle: Vehicle
    client: carla.Client
    world: carla.World
    vehicle_state: UnifiedState
    vehicle_action: UnifiedAction
    action_smoothing: float

    def __init__(self, vehicle, client, vehicle_state, vehicle_action, action_smoothing):
        self.vehicle = vehicle
        self.client = client
        self.vehicle_state = vehicle_state
        self.vehicle_action = vehicle_action
        self.action_smoothing = action_smoothing

        self.action_space = gym.spaces.Box(low=self.vehicle_action.action_lower_bound,
                                           high=self.vehicle_action.action_upper_bound,
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.vehicle_state.state_lower_bound,
                                                high=self.vehicle_state.state_upper_bound,
                                                dtype=np.float32)

    def step(self, action):
        if action is not None:
            u1, d1 = action[0], action[1]
            u2, d2 = action[2], action[3]
            u3, d3 = action[4], action[5]
            steer = np.random.normal(loc=u1, scale=np.sqrt(d1))
            throttle = np.random.normal(loc=u2, scale=np.sqrt(d2))
            brake = np.random.normal(loc=u3, scale=np.sqrt(d2))
            self.vehicle.control.steer = steer
            self.vehicle.control.throttle = throttle
            self.vehicle.control.brake = brake
            self.vehicle.tick()
            new_state = self.vehicle.get_state()
            done = False
        return new_state, 0, done, False, {}


class UniformPedestrianEnv(gym.Env):
    walker: carla.Walker
    client: carla.Client
    world: carla.World
    pedestrian_state: UnifiedState
    pedestrian_action: UnifiedAction

    def __init__(self, walker, client, world, pedestrian_state, pedestrian_action):
        self.walker = walker
        self.client = client
        self.world = world
        self.pedestrian_state = pedestrian_state
        self.pedestrian_action = pedestrian_action

        self.action_space = gym.spaces.Box(high=self.pedestrian_action.action_upper_bound,
                                           low=self.pedestrian_action.action_lower_bound,
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(high=self.pedestrian_state.state_upper_bound,
                                                low=self.pedestrian_state.state_lower_bound,
                                                dtype=np.float32)

    def step(self, action):
        if action is not None:
            u1, d1 = action[0], action[1]


        return














