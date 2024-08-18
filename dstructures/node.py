import utils.globalvalues as gv
import copy
import itertools
from utils.extendmath import d_distance_of_lon_action
from envs.world import CarModule


class Node(CarModule):
    """
    节点: 包含一个场景的驾驶环境
    """

    def __init__(self, ego_id, main_id, map, virtual_vehicle_dict) -> None:
        super().__init__(ego_id, main_id, map)
        self._virtual_vehicle_dict = virtual_vehicle_dict  # 一般只考虑距离较近的车辆


class Leaf(Node):
    """
    叶子节点: 包含下一时刻的驾驶环境
    """

    def __init__(self, ego_id, main_id, map, virtual_vehicle_dict) -> None:
        super().__init__(ego_id, main_id, map, virtual_vehicle_dict)
        self._risk = 0

    def clone_self(self):
        """
        弥补deepcopy无法作用于carla中的某些数据类型
        """
        new_vvehicle_dict = {}
        for vid in self._virtual_vehicle_dict.keys():
            new_vvehicle_dict[vid] = self._virtual_vehicle_dict[vid].clone_self()
        return Leaf(
            self._ego_id,
            self._main_id,
            self._map,
            new_vvehicle_dict,
        )

    def update_dict(self, dict):
        new_vvehicle_dict = {}
        for vid in dict.keys():
            new_vvehicle_dict[vid] = dict[vid].clone_self()
        return Leaf(
            self._ego_id,
            self._main_id,
            self._map,
            new_vvehicle_dict,
        )


class CIPORoot(Node):
    """
    根节点: 包含当前时间驾驶环境
    """

    def __init__(
        self, ego_id, main_id, map, virtual_vehicle_dict, lon_levels, lat_levels
    ) -> None:
        super().__init__(ego_id, main_id, map, virtual_vehicle_dict)

        self._lon_levels = lon_levels
        self._lat_levels = lat_levels

    def clone_self(self):
        """
        弥补deepcopy无法作用于carla中的某些数据类型
        """
        new_vvehicle_dict = {}
        for vid in self._virtual_vehicle_dict.keys():
            new_vvehicle_dict[vid] = self._virtual_vehicle_dict[vid].clone_self()
        return CIPORoot(
            self._ego_id,
            self._main_id,
            self._map,
            new_vvehicle_dict,
            self._lon_levels,
            self._lat_levels,
        )

    def generate_leaves_full(self):
        """
        全展开生成叶子节点
        返回所有叶子节点与其数量
        """
        leaves = []
        v_vehicle_id_list = list(self._virtual_vehicle_dict.keys())
        iter_list = self.generate_iter_list_full()
        for comb in iter_list:
            virtual_vehicle_dict_leaf = {}
            for i in range(len(comb)):
                v_action = comb[i]
                vid = v_vehicle_id_list[i]
                v_vehicle = self._virtual_vehicle_dict.get(vid)
                virtual_vehicle_dict_leaf[vid] = (
                    self.generate_next_step_virtual_vehicle(v_vehicle, v_action)
                )
            leaves.append(
                Leaf(self._ego_id, self._main_id, self._map, virtual_vehicle_dict_leaf)
            )
            return leaves

    def generate_leaves(self, dir_type):
        """
        生成叶子节点
        dir_type = "longitude" 或 "lateral"
        返回所有叶子节点与其数量
        """
        leaves = []
        iter_list = self.generate_iter_list(
            dir_type
        )  # dir_type决定叶子结点是纵向还是横向
        v_vehicle_id_list = list(self._virtual_vehicle_dict.keys())
        # 遍历每一个action的组合
        for comb in iter_list:
            virtual_vehicle_dict_leaf = {}
            for i in range(len(comb)):
                # 此处保证了v_action与vid是一一对应的
                v_action = comb[i]
                vid = v_vehicle_id_list[i]
                v_vehicle = self._virtual_vehicle_dict.get(vid)
                virtual_vehicle_dict_leaf[vid] = (
                    self.generate_next_step_virtual_vehicle(v_vehicle, v_action)
                )
            leaves.append(
                Leaf(self._ego_id, self._main_id, self._map, virtual_vehicle_dict_leaf)
            )
        return leaves, len(leaves)

    def generate_iter_list_full(self):
        res = []
        consider_actions = []
        if not self._virtual_vehicle_dict:
            return res
        for vid in self._virtual_vehicle_dict.keys():
            v_vehicle = self._virtual_vehicle_dict.get(vid)
            consider_action = gv.ACTION_SPACE
            if v_vehicle._waypoint.lane_id == gv.LANE_ID["Left"]:
                consider_action = [
                    "MAINTAIN",
                    "ACCELERATE",
                    "DECELERATE",
                    "SLIDE_RIGHT",
                ]
            if v_vehicle._waypoint.lane_id == gv.LANE_ID["Right"]:
                consider_action = ["MAINTAIN", "ACCELERATE", "DECELERATE", "SLIDE_LEFT"]
            consider_actions.append(consider_action)
        res = list(itertools.product(*consider_actions))
        return res

    def generate_iter_list(self, dir_type):
        """
        由于每个车辆所考虑的动作空间不同, 该方法可以返回不定长的动作空间的排列组合。
        dir_type = "longitude" 或 "lateral"
        """
        res = []
        consider_actions = []
        if not self._virtual_vehicle_dict:
            return res
        # 纵向动作
        if dir_type == "longitude":
            for vid in self._virtual_vehicle_dict.keys():
                v_vehicle = self._virtual_vehicle_dict.get(vid)
                # 自身
                if vid == self._ego_id:
                    consider_actions = ["MAINTAIN", "ACCELERATE", "DECELERATE"]
                elif vid in self._lon_levels["Level1"]:
                    consider_actions = ["MAINTAIN", "DECELERATE"]
                elif vid in self._lon_levels["Level2"]:
                    # 两车道限制
                    if v_vehicle._waypoint.lane_id == gv.LANE_ID["Left"]:
                        consider_actions = ["MAINTAIN", "SLIDE_RIGHT"]
                    if v_vehicle._waypoint.lane_id == gv.LANE_ID["Right"]:
                        consider_actions = ["MAINTAIN", "SLIDE_LEFT"]
                elif vid in self._lon_levels["Level3"]:
                    consider_actions = ["MAINTAIN"]
                else:
                    consider_actions = ["MAINTAIN"]
                res.append(consider_actions)
        # 横向动作
        if dir_type == "lateral":
            for vid in self._virtual_vehicle_dict.keys():
                v_vehicle = self._virtual_vehicle_dict.get(vid)
                if vid == self._ego_id:
                    # 两车道限制
                    if v_vehicle._waypoint.lane_id == gv.LANE_ID["Left"]:
                        consider_actions = ["SLIDE_RIGHT"]
                    if v_vehicle._waypoint.lane_id == gv.LANE_ID["Right"]:
                        consider_actions = ["SLIDE_LEFT"]
                elif vid == self._lat_levels["Level1"][0]:
                    # 在主车的侧前方
                    consider_actions = ["MAINTAIN", "DECELERATE"]
                elif vid == self._lat_levels["Level1"][1]:
                    # 在主车的侧后方
                    consider_actions = ["MAINTAIN", "ACCELERATE"]
                elif vid in self._lat_levels["Level2"]:
                    # 两车道限制
                    if v_vehicle._waypoint.lane_id == gv.LANE_ID["Left"]:
                        consider_actions = ["MAINTAIN", "SLIDE_RIGHT"]
                    if v_vehicle._waypoint.lane_id == gv.LANE_ID["Right"]:
                        consider_actions = ["MAINTAIN", "SLIDE_LEFT"]
                elif vid in self._lat_levels["Level3"]:
                    consider_actions = ["MAINTAIN"]
                else:
                    consider_actions = ["MAINTAIN"]
                res.append(consider_actions)
        # print(self._ego_id, list(itertools.product(*res)))
        return list(itertools.product(*res))

    def generate_next_step_virtual_vehicle(self, virtual_vehicle, control_action):
        """
        生成下一时刻的virtual vehicle
        """
        virtual_vehicle_next = copy.copy(virtual_vehicle)
        virtual_vehicle_next._control_action = control_action
        if control_action in ["MAINTAIN", "ACCELERATE", "DECELERATE"]:
            # 需要计算离散时间下的前进距离
            d_distance = d_distance_of_lon_action(virtual_vehicle, control_action, 1)
            virtual_vehicle_next._waypoint = virtual_vehicle._waypoint.next(d_distance)[
                0
            ]
        if control_action == "SLIDE_LEFT":
            d_distance = max(virtual_vehicle._scalar_velocity * 1, 1e-9)
            # 左侧道路上的路点
            virtual_vehicle_next._waypoint = virtual_vehicle._waypoint.next(d_distance)[
                0
            ].get_left_lane()
            if virtual_vehicle_next._waypoint is None:
                print(virtual_vehicle._waypoint.lane_id, "应该是-3")
        if control_action == "SLIDE_RIGHT":
            d_distance = max(virtual_vehicle._scalar_velocity * 1, 1e-9)
            # 右侧道路上的路点
            virtual_vehicle_next._waypoint = virtual_vehicle._waypoint.next(d_distance)[
                0
            ].get_right_lane()
            if virtual_vehicle_next._waypoint is None:
                print(virtual_vehicle._waypoint.lane_id, "应该是-2")
        virtual_vehicle_next._transform = virtual_vehicle_next._waypoint.transform
        virtual_vehicle_next._scalar_velocity = (
            virtual_vehicle._scalar_velocity + gv.LON_ACC_DICT.get(control_action)
        )
        return virtual_vehicle_next
