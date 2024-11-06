
from UniformEnv import UniformVehicleEnv
from PythonAPI.LLM2CarlaScenario.Wrappers.wrappers import *
from PythonAPI.LLM2CarlaScenario.State.BehaviorDefines import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class TrainingParameters:
    learning_rate = 0.01
    verbose = 0
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lamda = 0.95
    clip_range = 0.2
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    total_timesteps = 1000000


class TrainingCallBack(CheckpointCallback):
    def __init__(self, save_freq=20000, save_path='../models'):
        super(TrainingCallBack, self).__init__(save_freq=save_freq, save_path=save_path)

    def _on_rollout_end(self) -> None:
        obs = self.locals['obs']
        print(type(obs))
        print(len(obs))





class TrainingClient:
    client: carla.Client
    world: World
    ego_vehicle: Vehicle
    vehicle_evn: UniformVehicleEnv
    model_: PPO
    training_parameters: TrainingParameters
    training_callback: TrainingCallBack
    training_behavior: VehicleBehaviorBase

    def __init__(self, host="127.0.0.1", port=2000, training_parameter=None, training_behavior=None):
        try:
            self.client = carla.Client(host, port)
            self.world = World(self.client)
            self.ego_vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[12])
            self.vehicle_evn = UniformVehicleEnv(vehicle=self.ego_vehicle,
                                                 client=self.client,
                                                 vehicle_state=None,
                                                 vehicle_action=None,
                                                 action_smoothing=0.8)
            self.training_parameters = training_parameter
            self.model_ = PPO('MlpPolicy', env=self.vehicle_evn)
            self.training_behavior = training_behavior
            self.training_callback = TrainingCallBack(save_freq=20000, save_path=f'../models/{self.training_behavior.BehaviorName}')

        except Exception as e:
            if self.world is not None:
                print("error1")
                self.world.destroy()
            raise e
        finally:
            print("fin")

    def run(self):
        while True:
            self.spectator_look_at()
            self.world.tick()

    def training(self):
        self.model_.learn(total_timesteps=self.training_parameters.total_timesteps, callback=self.training_callback)


    def spectator_look_at(self):
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.actor.get_transform()
        new_transform = carla.Transform(
            transform.location + transform.get_forward_vector() * -10 + transform.get_up_vector() * 3,
            transform.rotation)
        spectator.set_transform(new_transform)


if __name__ == "__main__":
    training_parameters = TrainingParameters()
    b_lane_keeping = VehicleBehaviorLaneKeeping.default_behavior()
    training_client = TrainingClient(host='127.0.0.1', port=2000, training_behavior=b_lane_keeping)
    training_client.training()
