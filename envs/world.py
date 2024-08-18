# 好像没啥用
class World:
    """用于在全局获取各种信息"""

    def __init__(self, real_vehicle_dict) -> None:
        self._real_vehicle_dict = real_vehicle_dict

    def _add(self, id, real_vehicle):
        """添加键值对"""
        self._real_vehicle_dict[id] = real_vehicle

    def get_real_vehicle(self, id):
        """根据id获取车辆实例化"""
        return self._real_vehicle_dict.get(id)


class CarModule:
    """
    保存一些需要的车辆信息, 用于区分各项组件的所属
    """

    def __init__(self, ego_id, main_id, map) -> None:
        self._ego_id = ego_id
        self._main_id = main_id
        self._map = map
