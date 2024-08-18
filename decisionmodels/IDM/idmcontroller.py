import utils.extendmath as emath
import utils.globalvalues as gv
import math
import numpy as np
from decisionmodels.CDM.observer import Observer


class IntelligentDriverModel:
    """
    IDM Model, 一种基于加速度的智能驾驶员决策模型
    """

    def __init__(
        self,
        observer,
        time_gap,
        desired_speed,
        max_acceleration,
        minimum_gap,
        acc_exp,
        comf_deceleration,
        politness,
    ):
        self._observer = observer
        self._time_gap = time_gap  # 安全时距
        self._desired_speed = desired_speed  # 期望速度
        self._max_acceleration = max_acceleration  # 最大加速度
        self._minimum_gap = minimum_gap  # 能接受的与前车的最小距离
        self._acc_exp = acc_exp  # 公式里的δ
        self._comf_deceleration = comf_deceleration  # 舒适减速度
        self._control_acceleration = 0  # 最终输出的加速度
        self._politness = politness  # 责任系数

    def get_control_acceleration(self, front_dist, ego_v, delta_v):
        """
        计算下一步的加速度（纵向）
        """
        MINI_speed = 60 / 3.6
        if delta_v:
            # Desired Gap
            front_dist = max(front_dist, 0.01)
            s_star = self._minimum_gap + max(
                0,
                (
                    ego_v * self._time_gap
                    + (ego_v * delta_v)
                    / (
                        2
                        * math.sqrt(
                            abs(self._max_acceleration * self._comf_deceleration)
                        )
                    )
                ),
            )
            # Control Acceleration
            control_acceleration = self._max_acceleration * (
                1
                - math.pow(ego_v / self._desired_speed, self._acc_exp)
                - (math.pow(s_star, 2) / front_dist)
            )
        else:
            # Control Acceleration
            control_acceleration = self._max_acceleration * (
                1 - math.pow(ego_v / self._desired_speed, self._acc_exp)
            )

        if ego_v + control_acceleration <= MINI_speed:
            control_acceleration = (MINI_speed - ego_v) * gv.DECISION_DT

        return control_acceleration

    def run_forward(self, full_vehicle_id_dict, nnet=None):
        (
            front_id,
            front_dist,
            ego_v,
            delta_v,
            back_id,
            back_dist,
            sidef_id,
            sidef_dist,
            sideb_id,
            sideb_dist,
        ) = self._observer.observe(full_vehicle_id_dict)
        # 变道阈值
        LCthreshold = 2.5

        # Control Acceleration
        self._control_acceleration = np.clip(
            self.get_control_acceleration(front_dist, ego_v, delta_v),
            self._comf_deceleration,
            self._max_acceleration,
        )

        # 变道判断
        # 旁边有车无法变道
        if sidef_dist <= 0 or sideb_dist >= 0:
            return self._control_acceleration

        # 自身变道后加速度收益
        if sidef_id:
            ego_lc_delta_v = ego_v - full_vehicle_id_dict.get(sidef_id)._scalar_velocity

        else:
            sidef_dist = ego_lc_delta_v = None
        benefit_ego_acc = (
            np.clip(
                self.get_control_acceleration(sidef_dist, ego_v, ego_lc_delta_v),
                self._comf_deceleration,
                self._max_acceleration,
            )
            - self._control_acceleration
        )

        # 后车加速度收益
        if back_id:
            back_v = full_vehicle_id_dict.get(back_id)._scalar_velocity
            back_delta_v = back_v - ego_v
            back_acc = np.clip(
                self.get_control_acceleration(abs(back_dist), back_v, back_delta_v),
                self._comf_deceleration,
                self._max_acceleration,
            )
            if front_id:
                back_lc_delta_v = (
                    back_v - full_vehicle_id_dict.get(front_id)._scalar_velocity
                )
                back_lc_dist = abs(front_dist) + abs(back_dist) + gv.CAR_LENGTH
            else:
                back_lc_delta_v = back_lc_dist = None
            back_lc_acc = np.clip(
                self.get_control_acceleration(back_lc_dist, back_v, back_lc_delta_v),
                self._comf_deceleration,
                self._max_acceleration,
            )
            benefit_back_acc = back_lc_acc - back_acc
        else:
            benefit_back_acc = 0

        # 侧后车加速度收益
        if sideb_id:
            sideb_v = full_vehicle_id_dict.get(sideb_id)._scalar_velocity
            if sidef_id:
                sideb_delta_v = (
                    sideb_v - full_vehicle_id_dict.get(sidef_id)._scalar_velocity
                )
                sideb_oi_dist = abs(sidef_dist) + abs(sideb_dist) + gv.CAR_LENGTH
            else:
                sideb_delta_v = sideb_oi_dist = None
            sideb_acc = np.clip(
                self.get_control_acceleration(sideb_oi_dist, sideb_v, sideb_delta_v),
                self._comf_deceleration,
                self._max_acceleration,
            )
            sideb_lc_delta_v = sideb_v - ego_v
            sideb_lc_dist = abs(sideb_dist)
            sideb_lc_acc = np.clip(
                self.get_control_acceleration(sideb_lc_dist, sideb_v, sideb_lc_delta_v),
                self._comf_deceleration,
                self._max_acceleration,
            )
            benefit_sideb_acc = sideb_lc_acc - sideb_acc
        else:
            benefit_sideb_acc = 0

        # 计算变道与否
        if (
            benefit_ego_acc + self._politness * (benefit_back_acc + benefit_sideb_acc)
            > LCthreshold
        ):
            return "SLIDE"
        return self._control_acceleration


class IDMObserver(Observer):
    def __init__(self, ego_id, main_id, map) -> None:
        super().__init__(ego_id, main_id, map)

    def observe(self, vehicle_dict, mode="real"):
        """
        得到计算控制信息的参数(前车距离, 自身速度与前车速度差)
        """
        (
            front_id,
            front_dist,
            back_id,
            back_dist,
            sidef_id,
            sidef_dist,
            sideb_id,
            sideb_dist,
        ) = self.get_closest_vehicles(vehicle_dict, mode)
        # 前车距离保险杠的距离
        front_dist -= gv.CAR_LENGTH
        back_dist += gv.CAR_LENGTH
        sidef_dist -= gv.CAR_LENGTH
        sideb_dist += gv.CAR_LENGTH
        # 与前车速度差
        ego_vehicle = vehicle_dict.get(self._ego_id)
        front_vehicle = vehicle_dict.get(front_id)
        # 前方无车
        if not front_vehicle:
            front_dist = delta_v = None
        else:
            delta_v = ego_vehicle._scalar_velocity - front_vehicle._scalar_velocity

        return (
            front_id,
            front_dist,
            ego_vehicle._scalar_velocity,
            delta_v,
            back_id,
            back_dist,
            sidef_id,
            sidef_dist,
            sideb_id,
            sideb_dist,
        )

    def get_closest_vehicles(self, vehicle_dict, mode="real"):
        """
        筛选出主车前后侧方最近车辆
        mode = real || virtual, 代表传入的字典是realvehicle还是virtualvehicle
        """
        self._close_vehicle_id_list = self.get_close_vehicle_id_list(vehicle_dict)
        ego_vehicle = vehicle_dict.get(self._ego_id)
        min_dist_front, min_dist_back, min_dist_sidef, min_dist_sideb = (
            1e9,
            -1e9,
            1e9,
            -1e9,
        )
        min_id_front, min_id_back, min_id_sidef, min_id_sideb = None, None, None, None
        if mode == "real":
            ego_wp = self._map.get_waypoint(ego_vehicle._vehicle.get_location())
        elif mode == "virtual":
            ego_wp = ego_vehicle._waypoint
        else:
            raise ValueError("The mode value is wrong!")
        # 对每辆车判断是否处于同一车道且前后距离最短
        for rvid in self._close_vehicle_id_list:
            if rvid != self._ego_id:
                vehicle = vehicle_dict.get(rvid)
                if mode == "real":
                    veh_wp = self._map.get_waypoint(vehicle._vehicle.get_location())
                elif mode == "virtual":
                    veh_wp = vehicle._waypoint
                else:
                    raise ValueError("The mode value is wrong!")
                rel_distance = emath.cal_distance_along_road(ego_wp, veh_wp)
                if veh_wp.lane_id == ego_wp.lane_id:
                    # 前方车辆
                    if 0 < rel_distance < min_dist_front:
                        min_dist_front = rel_distance
                        min_id_front = rvid
                    # 后方车辆
                    if min_dist_back < rel_distance < 0:
                        min_dist_back = rel_distance
                        min_id_back = rvid
                if veh_wp.lane_id != ego_wp.lane_id:
                    # 侧前方车辆
                    if 0 < rel_distance < min_dist_front:
                        min_dist_sidef = rel_distance
                        min_id_sidef = rvid
                    # 侧后方车辆
                    if min_dist_sideb < rel_distance < 0:
                        min_dist_sideb = rel_distance
                        min_id_sideb = rvid
        # 返回前方最近车辆与其相对距离
        return (
            min_id_front,
            min_dist_front,
            min_id_back,
            min_dist_back,
            min_id_sidef,
            min_dist_sidef,
            min_id_sideb,
            min_dist_sideb,
        )
