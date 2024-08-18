import copy
import utils.extendmath as emath
import utils.globalvalues as gv


class FiniteStateMachine:
    """
    通过位置矩阵来判断危险场景
    """

    def __init__(self, step, pos_matirx, max_seq_len=3) -> None:
        self._state_list = [
            "Safe",
            "Overtake",
            "Overtaken",
            "Cutin",
            "RearEndRisk",
            "RearEndedRisk",
        ]
        self._current_step = step
        self._max_seq_len = max_seq_len
        self._pos_matrix = pos_matirx
        self._state_seq_to_main = {}
        self._last_seq_map = {}

    def __del__(self):
        pass

    def update(self, full_realvehicle_id_dict, step):
        """
        迭代更新, 判断主车车辆是否处于某种危险状态, 并判断具体的状态(状态之间并不严格冲突, 例如在切入的过程中也有可能产生追尾风险)
        """
        # 危险状态数量
        num_unsafe = 0
        # 现有的状态需要最长三个状态来判断
        assert self._max_seq_len >= 3
        # 时间更新
        self._current_step = step
        # Position Matrix更新
        self._last_seq_map = copy.deepcopy(self._state_seq_to_main)
        position_map, related_vehicles = self._pos_matrix.state_update(
            full_realvehicle_id_dict
        )
        # 更新一对一的状态序列
        # state_seq_to_main格式如下:
        # Key为与被测车交互的车辆id, 对应的Value是一个长度最长为max_seq_len的列表
        # 列表中的元素是四元组: (在posmatrix中所属的key, 主车车道id, 当前车道id, 最后处于该状态的step值)
        for rvid in related_vehicles:
            position_key = self._pos_matrix.get_key(position_map, rvid)
            if position_key == None:
                continue
            if rvid not in self._state_seq_to_main.keys():
                self._state_seq_to_main[rvid] = list()
            laneid = self._pos_matrix._map.get_waypoint(
                full_realvehicle_id_dict.get(rvid)._vehicle.get_location()
            ).lane_id
            lane_id_main = self._pos_matrix._map.get_waypoint(
                full_realvehicle_id_dict.get(
                    self._pos_matrix._main_id
                )._vehicle.get_location()
            ).lane_id
            # 构建三元组
            seq_elem_tuple = (position_key, lane_id_main, laneid, self._current_step)
            # 序列为空
            if not self._state_seq_to_main[rvid]:
                self._state_seq_to_main[rvid].append(seq_elem_tuple)
            # 如果序列的最后一个元素与当前状态相同, 直接原地修改
            elif position_key == self._state_seq_to_main[rvid][-1][0]:
                self._state_seq_to_main[rvid][-1] = seq_elem_tuple
            else:
                # 超出max_seq_len, 控制列表长度
                if len(self._state_seq_to_main[rvid]) > self._max_seq_len:
                    self._state_seq_to_main[rvid].pop(0)
                self._state_seq_to_main[rvid].append(seq_elem_tuple)
        # 判断是否处于某种危险状态
        for rvid in related_vehicles:
            # 对每个state都做一次条件判断
            for state in self._state_list:
                if self.state_condition(rvid, full_realvehicle_id_dict, state):
                    print(
                        f"危险发生在场景开始后的第{int(step * gv.STEP_DT)}秒",
                        "交互车辆为: ",
                        rvid,
                        " 状态为: ",
                        state,
                    )
                    num_unsafe += 1
        return num_unsafe

    def state_condition(self, rvid, full_realvehicle_id_dict, state="Safe"):
        """
        根据状态序列来判断状态
        """
        ttc_threhold = 3
        seq = self._state_seq_to_main.get(rvid)
        last_seq = self._last_seq_map.get(rvid)
        if not seq:
            return False
        # 当前状态是否为最新
        latest_condition = seq[-1][-1] == self._current_step
        # 序列是否重复
        if seq and last_seq:
            repeat_condition = self.non_repeat_condition(seq, last_seq)
        else:
            repeat_condition = True
        if state == "Cutin":
            # 需要满足的基本条件
            basic_condition = (
                repeat_condition
                and self.seq_lane_limit_condition(seq, 2)
                and latest_condition
                and self.keep_lane_condition(seq, 2, True)
            )
            # 切入的判断条件
            core_condition = (
                len(seq) >= 2
                and seq[-1][0] == "Front"
                and seq[-2][0]
                in [
                    "LeftFront",
                    "RightFront",
                    "LeftSide",
                    "RightSide",
                ]
            )
            return basic_condition and core_condition
        if state == "Overtaken":
            # 需要满足的基本条件
            basic_condition = (
                repeat_condition
                and self.seq_lane_limit_condition(seq, 3)
                and latest_condition
            )
            # 被超车的判断条件
            core_condition = (
                len(seq) >= 3
                and seq[-1][0] in ["Front", "LeftFront", "RightFront"]
                and seq[-3][0]
                in [
                    "LeftBack",
                    "RightBack",
                    "Back",
                ]
            )
            return basic_condition and core_condition
        if state == "Overtake":
            # 需要满足的基本条件
            basic_condition = (
                repeat_condition
                and self.seq_lane_limit_condition(seq, 3)
                and latest_condition
            )
            # 超车的判断条件
            core_condition = (
                len(seq) >= 3
                and seq[-1][0]
                in [
                    "LeftBack",
                    "RightBack",
                    "Back",
                ]
                and seq[-3][0] in ["Front", "LeftFront", "RightFront"]
            )
            return basic_condition and core_condition
        if state in ["RearEndRisk", "RearEndedRisk"]:
            min_id_front, min_dist_front, min_id_back, min_dist_back = (
                self.get_lon_closest_vehicle(
                    full_realvehicle_id_dict,
                    self._pos_matrix._related_vehicles,
                    self._pos_matrix._map,
                    self._pos_matrix._main_id,
                )
            )
            if state == "RearEndRisk":
                front_ttc = -1e9
                if min_id_front:
                    front_ttc = min_dist_front / (
                        full_realvehicle_id_dict.get(
                            self._pos_matrix._main_id
                        )._scalar_velocity
                        - full_realvehicle_id_dict.get(min_id_front)._scalar_velocity
                        + 1e-9
                    )
                return front_ttc > 0 and front_ttc <= ttc_threhold
            if state == "RearEndedRisk":
                back_ttc = -1e9
                if min_id_back:
                    back_ttc = -min_dist_back / (
                        full_realvehicle_id_dict.get(min_id_back)._scalar_velocity
                        - full_realvehicle_id_dict.get(
                            self._pos_matrix._main_id
                        )._scalar_velocity
                        + 1e-9
                    )
                return back_ttc > 0 and back_ttc <= ttc_threhold

    @staticmethod
    def non_repeat_condition(seq, last_seq):
        """
        根据前一序列的相同与否, 判断是否是重复状态
        """
        if len(seq) <= 1 or len(last_seq) <= 1:
            return True
        return seq[:-2] != last_seq[:-2]

    @staticmethod
    def seq_lane_limit_condition(seq, limit_len):
        """
        判断squence的长度是否符合
        """
        return len(seq) >= limit_len

    @staticmethod
    def keep_lane_condition(seq, search_len, is_main):
        """
        判断是否保持车道, search_len表示搜索的状态时间长度, is_main表示是否看主车, False则看当前车辆
        """
        assert search_len > 0
        seq_len = len(seq)
        if search_len <= 1:
            return True
        real_search_len = min(search_len, seq_len)
        if is_main:
            laneid_id = 1
        else:
            laneid_id = 2
        std_laneid = seq[-1][laneid_id]
        for i in range(seq_len - real_search_len, seq_len - 1):
            if seq[i][laneid_id] != std_laneid:
                return False
        return True

    @staticmethod
    def get_lon_closest_vehicle(real_vehicle_dict, rel_vehicle_list, map, ego_id):
        """
        筛选出主车前后方最近车辆
        """
        ego_vehicle = real_vehicle_dict.get(ego_id)
        min_dist_front = 1e9
        min_dist_back = -1e9
        min_id_front = None
        min_id_back = None
        ego_wp = map.get_waypoint(ego_vehicle._vehicle.get_location())
        # 对每辆车判断是否处于同一车道且前后距离最短
        for rvid in rel_vehicle_list:
            if rvid != ego_id:
                vehicle = real_vehicle_dict.get(rvid)
                veh_wp = map.get_waypoint(vehicle._vehicle.get_location())
                rel_distance = emath.cal_distance_along_road(ego_wp, veh_wp)
                # 前方
                if veh_wp.lane_id == ego_wp.lane_id:
                    # 前方车辆
                    if 0 < rel_distance < min_dist_front:
                        min_dist_front = rel_distance
                        min_id_front = rvid
                    # 后方车辆
                    if min_dist_back < rel_distance < 0:
                        min_dist_back = rel_distance
                        min_id_back = rvid
        # 返回前后方最近车辆与其相对距离
        return min_id_front, min_dist_front, min_id_back, min_dist_back
