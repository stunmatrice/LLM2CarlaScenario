
import carla
from UniformEnv import UniformVehicleEnv
from PythonAPI.LLM2CarlaScenario.Wrappers.wrappers import *

class TrainingClient():
    client: carla.Client
    world: World
    egoVehicle: Vehicle
    vehileEvn: UniformVehicleEnv
    def __init__(self, host="127.0.0.1", port=2000):
        try:
            self.client = carla.Client(host, port)
            self.world = World(self.client)
            self.egoVehicle = Vehicle(self.world, self.world.map.get_spawn_points()[12])
            self.vehileEvn = UniformVehicleEnv(vehicle=self.egoVehicle, )

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

    def spectator_look_at(self):
        spectator = self.world.get_spectator()
        transform = self.egoVehicle.actor.get_transform()
        newTranform = carla.Transform(
            transform.location + transform.get_forward_vector() * -10 + transform.get_up_vector() * 3,
            transform.rotation)
        spectator.set_transform(newTranform)


if __name__ == "__main__":
    trainingClient = TrainingClient()
    trainingClient.run()