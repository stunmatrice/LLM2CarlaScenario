import carla
import random
import time
import collections
import math
import numpy as np
import weakref
import pygame


def print_transform(transform):
    print("Location(x={:.2f}, y={:.2f}, z={:.2f}) Rotation(pitch={:.2f}, yaw={:.2f}, roll={:.2f})".format(
        transform.location.x,
        transform.location.y,
        transform.location.z,
        transform.rotation.pitch,
        transform.rotation.yaw,
        transform.rotation.roll
    )
    )


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate - 1] + u"\u2026") if len(name) > truncate else name


def angle_diff(v0, v1):
    """ Calculates the signed angle difference (-pi, pi] between 2D vector v0 and v1 """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def distance_to_line(A, B, p):
    num = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom


def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])


camera_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7))
}


# ===============================================================================
# CarlaActorBase
# ===============================================================================

class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.world.actor_list.append(self)
        self.destroyed = False

    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            #print("Destroying ", self, "...")
            self.actor.destroy()
            self.world.actor_list.remove(self)
            self.destroyed = True

    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)


# ===============================================================================
# CollisionSensor
# ===============================================================================

class CollisionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_collision_fn):
        self.on_collision_fn = on_collision_fn

        # Collision history
        self.history = []

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.collision")

        # Create and setup sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_collision_fn
        if callable(self.on_collision_fn):
            self.on_collision_fn(event)


# ===============================================================================
# LaneInvasionSensor
# ===============================================================================

class LaneInvasionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_invasion_fn):
        self.on_invasion_fn = on_invasion_fn

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")

        # Create sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: LaneInvasionSensor.on_invasion(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_invasion_fn
        if callable(self.on_invasion_fn):
            self.on_invasion_fn(event)


# ===============================================================================
# Camera
# ===============================================================================

class Camera(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(),
                 sensor_tick=0.0, attach_to=None, on_recv_image=None,
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw):
        self.on_recv_image = on_recv_image
        self.color_converter = color_converter

        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.on_recv_image(array)

    def destroy(self):
        super().destroy()


# ===============================================================================
# Vehicle
# ===============================================================================

class Vehicle(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None,
                 vehicle_type="vehicle.lincoln.mkz"):
        # Setup vehicle blueprint
        # bp_lib = world.get_blueprint_library()
        # for bp in bp_lib:
        #     print(bp.id)

        vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        color = vehicle_bp.get_attribute("color").recommended_values[0]
        vehicle_bp.set_attribute("color", color)

        # Create vehicle actor
        actor = world.spawn_actor(vehicle_bp, transform)
        #print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl()
        self.spawn_point = transform
        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    def tick(self):
        self.actor.apply_control(self.control)

    def get_speed(self):
        velocity = self.actor.get_velocity()
        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2)
        #speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return speed

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.actor.get_transform().location, project_to_road=True)

    #def get_velocity(self):
        #return self.get_velocity()

    def get_d(self):
        way_point = self.get_closest_waypoint()
        delta = self.get_transform().location - way_point.transform.location
        return delta

    def reset_position(self):
        self.actor.set_transform(self.spawn_point)
        self.actor.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))
        # self.control = carla.VehicleControl()
        # self.control.throttle = 0.0  # 设置油门为0
        # self.control.brake = 1.0  # 设置刹车为1，确保车辆完全停止
        # self.tick()
        self.world.tick()
        #print(self.get_speed())

    # 状态空间的维度调整 4 - 14 ， 增加前方
    # dx dy 中心车道偏移
    # 车速大小
    # 11个路点的 方向点乘车速方向

    # 未使用全局坐标方向
    def get_state(self):
        wp = self.get_closest_waypoint()
        vehicle_transform = self.actor.get_transform()
        z = vehicle_transform.location.z
        vehicle_direction = vehicle_transform.get_forward_vector()
        d = vehicle_transform.location - wp.transform.location
        speed = self.get_speed()
        #print(speed)
        wps = [wp]
        c_wp = wp
        for i in range(10):
            next_wp = c_wp.next(2)[0]
            wps.append(next_wp)

        dps = []
        for i_wp in wps:
            wp_direction = i_wp.transform.get_forward_vector()
            dot_product = wp_direction.dot(vehicle_direction)
            dps.append(dot_product)
        dps.insert(0, d.x)
        dps.insert(1, d.y)
        dps.insert(2, speed)
        return np.array(dps, dtype=np.float32)

    def get_dot_product_group(self):
        wp = self.get_closest_waypoint()
        wps = wp.next(10)
        dps = []
        fy = 0.8
        for w in wps:
            wp_direction = w.transform.get_forward_vector()
            vehicle_transform = self.get_transform()
            vehicle_direction = vehicle_transform.get_forward_vector()
            dot_product = wp_direction.dot(vehicle_direction) * fy
            dps.append(dot_product)
            fy = fy * 0.8
        return dps


# ===============================================================================
# World
# ===============================================================================

class World():
    def __init__(self, client):
        self.world = client.get_world()
        self.map = self.get_map()
        self.actor_list = []

    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()
        self.world.tick()

    def destroy(self):
        #print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()



    def get_carla_world(self):
        return self.world

    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)


    def find_nearby_vehicles(self, ego):
        res = []
        for actor in self.actor_list:
            if isinstance(actor, Vehicle) and actor is not ego:
                location1 = ego.get_transform().location
                location2 = actor.get_transform().location
                dis = location1.distance(location2)

                forward = actor.get_transform().get_forward_vector()
                delta = location2 - location1
                dot = forward.x * delta.x + forward.y * delta.y + forward.z * delta.z
                if dis < 50 and dot > 0:
                    res.append(actor)
        return res