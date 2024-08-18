"""
我们假设初始化4辆车, 主车*1 + NPC车*2 + ACDM*1
"""

import gymnasium as gym
from gymnasium.spaces.box import Box
import carla
import numpy as np
from main import initialize


class ACDMInitEnv(gym.Env):
    def __init__(self):
        # Carla部分初始化
        (
            self.world,
            self.map,
            self.full_realvehicle_dict,
            self.adv_realvehicle_list,
            self.vehicle_bp,
            self.spectator,
            self.pred_net,
        ) = initialize()
        self.full_realvehicle_list = list(self.full_realvehicle_dict.keys())
        self.left_lane_id = -2
        self.right_lane_id = -3
        self.car_length = 4.791779518127441
        self.car_width = 2.163450002670288
        # 状态空间为7个维度: 主车:[初速度] + NPC1:[初速度, 相对车道, 相对距离] + NPC2:[初速度, 相对车道, 相对距离]
        # 将初始速度限制在20-30之间, 相对距离在50m以内, 相对车道为0(<0.5)表示不同车道, 为1(>=0.5)表示相同车道
        self.observation_space = Box(
            low=np.array([20, 20, 0, -50, 20, 0, -50]),
            high=np.array([30, 30, 1, 50, 30, 1, 50]),
        )
        # 动作空间为4个维度: ACDM:[初速度, 相对车道, 相对距离, 驾驶风格]
        # 驾驶风格保持20到100的限制
        self.action_space = Box(
            low=np.array([20, 0, -50, 20]), high=np.array([30, 1, 50, 100])
        )

    def step(self):
        pass

    def reset(self):
        while True:
            self.observation_space.sample()
            break
        for carid in self.full_realvehicle_list:
            self.full_realvehicle_dict[carid]._vehicle.destroy()
            del self.full_realvehicle_dict[carid]

    def render(self):
        return None

    def close(self):
        return None

    def judge_state_reasonable(self, state_box):
        car0_lane, car0_pos = state_box[2], state_box[3]
        car1_lane, car1_pos = state_box[5], state_box[6]
        # NPC与主车初始位置不可碰撞
        if (car0_lane < 0.5 and abs(car0_pos) <= self.car_length * 2) or (
            car1_lane < 0.5 and abs(car1_pos) <= self.car_length * 2
        ):
            return False
        # NPC之间不可碰撞
        if (car0_lane < 0.5 and car1_lane < 0.5) or (
            car0_lane >= 0.5 and car1_lane >= 0.5
        ):
            if abs(car0_pos - car1_pos) <= self.car_length * 2:
                return False
