import carla
import math
from utils.globalvalues import CAR_LENGTH, CAR_WIDTH


class VirtualVehicle:
    """
    代表未来状态的车辆, 不会出现在Carla中
    """

    def __init__(self, id, waypoint, transform, scalar_velocity, control_action):
        self._id = id
        self._waypoint = waypoint
        self._transform = transform
        self._scalar_velocity = scalar_velocity
        self._control_action = control_action

    def clone_self(self):
        return VirtualVehicle(
            self._id,
            self._waypoint,
            self._transform,
            self._scalar_velocity,
            self._control_action,
        )

    def judge_collision(self, other_virtual_vehicle):
        # 与realvehicle.py中的collision_callback基本相同
        if (
            self._transform.location == carla.Location(0, 0, 0)
            or self._transform.location.distance_2d(
                other_virtual_vehicle._transform.location
            )
            > 6
        ):
            return False
        if (
            self._transform.location == carla.Location(0, 0, 0)
            or self._transform.location.distance_2d(
                other_virtual_vehicle._transform.location
            )
            <= CAR_WIDTH
        ):
            return True
        distance_vector = (
            other_virtual_vehicle._transform.location - self._transform.location
        )
        forward_vector = self._transform.get_forward_vector()
        projection = abs(
            distance_vector.dot_2d(forward_vector)
            / forward_vector.distance_2d(carla.Vector3D(0, 0, 0))
        )
        if (
            distance_vector.distance_squared_2d(carla.Vector3D(0, 0, 0))
            - projection * projection
            > 0.01
        ):
            normal = math.sqrt(
                distance_vector.distance_squared_2d(carla.Vector3D(0, 0, 0))
                - projection * projection
            )
        else:
            normal = 0
        if projection <= CAR_LENGTH * 1 and normal <= CAR_WIDTH * 1.3:
            return True
        else:
            return False