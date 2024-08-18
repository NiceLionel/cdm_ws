import carla
import utils.globalvalues as gv
import math
import random
from vehicles.virtualvehicle import VirtualVehicle
from utils.extendmath import cal_yaw_by_location


class RealVehicle:
    """
    被生成在Carla中的实际车辆, 包含了额外的物理信息
    """

    def __init__(
        self,
        map,
        vehicle,
        spawn_point,
        controller,
        scalar_velocity=0,
        control_action="MAINTAIN",
    ):
        self._vehicle = vehicle
        self._spawn_point = (
            spawn_point  # carla初始location默认是0,0,0, 因此需要一个初始位置的输入
        )
        self._scalar_velocity = scalar_velocity
        self._control_action = control_action
        self._controller = controller
        self._map = map
        self._next_waypoint = None
        self._target_lane_id = None
        self._lane_changing_route = []
        self._changing_lane_pace = 0  # 处理变道用，判断是否在变道中以及完成了多少
        self._nade_observer = None  # 仅仅在D2RL对比实验中使用

    def __del__(self):
        pass

    def run_step(self, full_vehicle_dict, nnet=None):
        """一次循环"""
        self._control_action = self._controller.run_forward(full_vehicle_dict, nnet)

    def clone_to_virtual(self):
        """
        创建一个状态与自身相同的VirtualVehicle
        """
        return VirtualVehicle(
            self._vehicle.id,
            self._map.get_waypoint(self._vehicle.get_location()),
            self._vehicle.get_transform(),
            self._scalar_velocity,
            self._control_action,
        )

    def cal_lanechanging_route(self):
        """
        计算变道过程中的路径
        Return: List[carla.Transform]
        """
        traj_length = int(gv.LANE_CHANGE_TIME / gv.STEP_DT)
        route = [None] * (traj_length + 1)
        # Route的首个元素包含变道开始前最后一个wp, 在调用时轨迹应从route[1]开始
        route[0] = self._vehicle.get_transform()
        route[-1] = self._target_dest_wp.transform
        # 计算变道时车辆的朝向
        direction_vector = route[-1].location - route[0].location
        yaw = cal_yaw_by_location(route[-1].location, route[0].location)
        rotation = carla.Rotation(pitch=0, yaw=yaw, roll=0)
        for i in range(1, traj_length):
            location = carla.Location(
                x=route[0].location.x + i * direction_vector.x / (traj_length - 1),
                y=route[0].location.y + i * direction_vector.y / (traj_length - 1),
                z=route[0].location.z,
            )
            wp = self._map.get_waypoint(location)
            location.z = wp.transform.location.z
            rotation.pitch = wp.transform.rotation.pitch
            route[i] = carla.Transform(
                location=location,
                rotation=rotation,
            )
        return route

    def descrete_control(self):
        """
        离散动作转换为路径规划
        """
        longitude_action_list = ["MAINTAIN", "ACCELERATE", "DECELERATE"]
        lateral_action_list = ["SLIDE_LEFT", "SLIDE_RIGHT", "SLIDE"]
        if self._next_waypoint == None:
            self._next_waypoint = self._map.get_waypoint(self._spawn_point.location)
        if (
            self._control_action in longitude_action_list
            or type(self._control_action) != str
        ):
            # 加速, 减速, 保持车道, 保持车道, 过程离散化
            if (
                self._scalar_velocity <= gv.MIN_SPEED
                and self._control_action == "DECELERATE"
            ):
                self._control_action = "MAINTAIN"
            if type(self._control_action) == str:
                self._scalar_velocity += (
                    gv.LON_ACC_DICT.get(self._control_action) * gv.STEP_DT
                )
            else:
                self._scalar_velocity += self._control_action * gv.STEP_DT
            wp_gap = self._scalar_velocity * gv.STEP_DT
            if wp_gap == 0:
                pass
            if wp_gap > 0:
                self._next_waypoint = self._next_waypoint.next(wp_gap)[0]
            if wp_gap < 0:
                self._next_waypoint = self._next_waypoint.previous(-wp_gap)[0]
            self._target_lane_id = self._next_waypoint.lane_id
            self._vehicle.set_transform(self._next_waypoint.transform)

        if self._control_action in lateral_action_list:
            # 涉及到车道变换的动作, 设定一个目标路点, 一秒行驶到目标位置
            current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
            self._current_dest_wp = current_waypoint.next(self._scalar_velocity)[0]
            if self._control_action == "SLIDE":
                if current_waypoint.lane_id == gv.LANE_ID["Left"]:
                    self._control_action = "SLIDE_RIGHT"
                if current_waypoint.lane_id == gv.LANE_ID["Right"]:
                    self._control_action = "SLIDE_LEFT"
            if self._control_action == "SLIDE_LEFT":
                if self._current_dest_wp.lane_id == gv.LANE_ID["Right"]:
                    # 目标车道存在
                    self._target_dest_wp = self._current_dest_wp.get_left_lane()
                else:
                    self._target_dest_wp = self._current_dest_wp
            if self._control_action == "SLIDE_RIGHT":
                if self._current_dest_wp.lane_id == gv.LANE_ID["Left"]:
                    # 目标车道存在
                    self._target_dest_wp = self._current_dest_wp.get_right_lane()
                else:
                    self._target_dest_wp = self._current_dest_wp
            if self._changing_lane_pace == 0:
                self._lane_changing_route = self.cal_lanechanging_route()
                self._target_lane_id = self._map.get_waypoint(
                    self._lane_changing_route[-1].location
                ).lane_id
            self._changing_lane_pace += 1
            if self._changing_lane_pace < len(self._lane_changing_route) - 1:
                self._next_waypoint = self._map.get_waypoint(
                    self._lane_changing_route[self._changing_lane_pace].location
                )
                self._vehicle.set_transform(
                    self._lane_changing_route[self._changing_lane_pace]
                )
            else:
                self._changing_lane_pace = 0
                self._next_waypoint = self._map.get_waypoint(
                    self._lane_changing_route[-1].location
                ).next(self._scalar_velocity * gv.STEP_DT)[0]
                self._vehicle.set_transform(self._next_waypoint.transform)

    def collision_callback(self, other_vehicle):
        """
        判断两车是否发生碰撞
        """
        # 若两车水平距离大于bounding box对角线长度之和，则必然未碰撞
        if (
            self._vehicle.get_location() == carla.Location(0, 0, 0)
            or self._vehicle.get_location().distance_squared_2d(
                other_vehicle.get_location()
            )
            > (
                self._vehicle.bounding_box.extent.x**2
                + self._vehicle.bounding_box.extent.y**2
            )
            * 4
        ):
            return False
        # 否则计算两车相对位置向量在主车前进方向上的投影，与车身长度进行比较
        distance_vector = other_vehicle.get_location() - self._vehicle.get_location()
        forward_vector = self._vehicle.get_transform().get_forward_vector()
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
        if (
            projection
            <= (
                self._vehicle.bounding_box.extent.x
                + other_vehicle.bounding_box.extent.x
            )
            * 0.9
            and normal
            <= (
                self._vehicle.bounding_box.extent.y
                + other_vehicle.bounding_box.extent.y
            )
            * 0.8
        ):
            return True
        else:
            return False
