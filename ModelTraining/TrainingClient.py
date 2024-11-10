import numpy as np

from UniformEnv import UniformVehicleEnv
from PythonAPI.LLM2CarlaScenario.Wrappers.wrappers import *
from PythonAPI.LLM2CarlaScenario.State.BehaviorDefines import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import signal

from stable_baselines3.common.buffers import RolloutBuffer
from PythonAPI.LLM2CarlaScenario.LLMTools.LLMAutoReward import LLMAutoReward

class TrainingParameters:
    learning_rate = 0.01
    verbose = 0
    n_steps = 1024
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lamda = 0.95
    clip_range = 0.2
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    total_timesteps = 1000000

    def __init__(self):
        self.learning_rate = 0.01
        self.verbose = 0
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lamda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.total_timesteps = 1000000


class TrainingCallBack(CheckpointCallback):
    def __init__(self, save_freq=20000, save_path='../models', training_client=None, behavior=None):
        super(TrainingCallBack, self).__init__(save_freq=save_freq, save_path=save_path)
        self.training_client = training_client
        self.behavior = behavior

    def _on_step(self) -> bool:
        # self.training_client.world.tick()
        self.training_client.spectator_look_at()
        return True

    # dict_keys(['self', 'total_timesteps', 'callback', 'log_interval', 'tb_log_name', 'reset_num_timesteps', 'progress_bar', 'iteration', 'env', 'rollout_buffer', 'n_rollout_steps', 'n_steps', 'obs_tensor', 'actions', 'values', 'log_probs', 'clipped_actions', 'new_obs', 'rewards', 'dones', 'infos', 'idx', 'done']
    def _on_rollout_end(self) -> None:
        rewards = self.locals['rewards']

        # print(self.locals.keys())
        print(self.globals.keys())
        print(type(rewards))
        print(rewards.size)
        rb:RolloutBuffer = self.locals['rollout_buffer']
        print(rb.rewards.shape)
        rb.rewards.fill(0)
        # 1 Get reference of state data and the behaviors prompt here
        # 2 Sent key frames of state data, let LLM decide where the curve is good
        ##
        behavior: VehicleBehaviorBase = self.behavior
        behavior_prompt = behavior.BehaviorPrompt
        print(rb.observations.shape)

        obs = rb.observations[0::90, 0, :]
        last = rb.observations[-1, 0, :]
        data = np.vstack([obs, last])
        data_2_str = np.array2string(data, precision=3, separator=',', suppress_small=True)
        print(data.shape)

        auto_reward = LLMAutoReward(system_prompt=LLMAutoReward.default_system_prompt(),
                                    behavior_prompt=behavior_prompt,
                                    data_prompt=data_2_str,
                                    output_format_prompt=LLMAutoReward.default_output_format_prompt())
        res = auto_reward.get_result()
        print(res)


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

            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1 / 60
            self.world.apply_settings(settings)

            self.ego_vehicle = Vehicle(world=self.world,
                                       transform=self.world.map.get_spawn_points()[7],
                                       on_collision_fn=self._on_collision_fn)

            # self.npc1 = Vehicle(world=self.world,
            #                     transform=self.world.map.get_spawn_points()[28])

            self.vehicle_evn = UniformVehicleEnv(vehicle=self.ego_vehicle,
                                                 client=self.client,
                                                 vehicle_state=None,
                                                 vehicle_action=None,
                                                 action_smoothing=0.8)
            self.training_parameters = training_parameter
            self.model_ = PPO('MlpPolicy',
                              env=self.vehicle_evn,
                              n_steps=1024)
            self.training_behavior = training_behavior
            self.training_callback = TrainingCallBack(save_freq=20000,
                                                      save_path=f'../models/{self.training_behavior.BehaviorName}',
                                                      training_client=self,
                                                      behavior=self.training_behavior)

            signal.signal(signal.SIGINT, self.handle_mannual_exit)
        except Exception as e:
            if self.world is not None:
                print("error1")
                self.world.destroy()
            raise e
        finally:
            print("fin")
            # if self.world is not None:
            #     print("error1")
            #     self.world.destroy()

    def run(self):
        while True:
            self.spectator_look_at()
            self.world.tick()

    def training(self):
        self.model_.learn(total_timesteps=self.training_parameters.total_timesteps,
                          callback=self.training_callback)

    def spectator_look_at(self):
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.actor.get_transform()
        new_transform = carla.Transform(
            transform.location + transform.get_forward_vector() * -10 + transform.get_up_vector() * 3,
            transform.rotation)
        spectator.set_transform(new_transform)

    def handle_mannual_exit(self, signum, frame):
        if self.world is not None:
            self.world.destroy()

    def _on_collision_fn(self, event):
        self.ego_vehicle.should_done = True



if __name__ == "__main__":
    training_parameters = TrainingParameters()
    b_lane_keeping = VehicleBehaviorLaneKeeping.default_behavior()
    training_client = TrainingClient(host='127.0.0.1',
                                     port=2000,
                                     training_parameter=training_parameters,
                                     training_behavior=b_lane_keeping)
    training_client.training()
