import utils.globalvalues as gv
import utils.extendmath as emath


class PositionMatrix:
    """
    以被测车辆为中心, 周围八个方向作栅格, 在栅格内的车辆id被保存至哈希表, 通过查表的方式定性确定相对位置。
    """

    def __init__(self, main_id, map) -> None:
        self._position_map = {}
        self._related_vehicles = []
        self._main_id = main_id
        self._map = map

    def __del__(self):
        pass

    def state_update(self, full_realvehicle_id_dict):
        self._position_map = {
            "Front": [],
            "LeftFront": [],
            "RightFront": [],
            "LeftSide": [],
            "RightSide": [],
            "Back": [],
            "LeftBack": [],
            "RightBack": [],
        }
        self._related_vehicles = []
        for rvid in full_realvehicle_id_dict.keys():
            if rvid != self._main_id:
                # 判断是否较近
                start_wp = self._map.get_waypoint(
                    full_realvehicle_id_dict.get(self._main_id)._vehicle.get_location()
                )
                target_wp = self._map.get_waypoint(
                    full_realvehicle_id_dict.get(rvid)._vehicle.get_location()
                )
                rel_distance = emath.cal_distance_along_road(start_wp, target_wp)
                if abs(rel_distance) > gv.OBSERVE_DISTANCE:
                    continue
                self._related_vehicles.append(rvid)
                # 近处车辆, 通过方位和车道判断位置
                main_laneid, other_laneid = start_wp.lane_id, target_wp.lane_id
                # 纵向标签
                lon_string = ""
                if rel_distance > gv.CAR_LENGTH:
                    lon_string = "Front"
                elif rel_distance < -gv.CAR_LENGTH:
                    lon_string = "Back"
                else:
                    lon_string = "Side"
                # 横向标签
                lat_string = ""
                if main_laneid > other_laneid:
                    lat_string = "Right"
                elif main_laneid < other_laneid:
                    lat_string = "Left"
                key = lat_string + lon_string
                # 避免仿真器的误差引起的判断错误
                if key == "Side":
                    key = "Front" if rel_distance > 0 else "Back"
                self._position_map[key].append(rvid)
        return self._position_map, self._related_vehicles

    @staticmethod
    def get_key(list_dict, vid):
        """
        由value中的元素得到key
        """
        for key, value in list_dict.items():
            if vid in value:
                return key

        return None
