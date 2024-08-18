from dstructures.node import CIPORoot
from envs.world import CarModule
import utils.globalvalues as NUM_OF_TREE


class EnumerateTree(CarModule):
    """
    包含根节点和叶子节点
    """

    def __init__(self, ego_id, main_id, map) -> None:
        super().__init__(ego_id, main_id, map)
        self._root = None
        self._leaves = []
        self._num_lon_leaves = 0
        self._num_lat_leaves = 0
        self._probability = 0
        self._is_valid = False

    def generate_root_from_cipo(
        self, real_vehicle_dict, close_vehicle_id_list, lon_levels, lat_levels
    ):
        """
        从CIPO Observer生成Root节点
        """
        virtual_vehicle_dict = {}
        for cid in close_vehicle_id_list:
            virtual_vehicle_dict[cid] = real_vehicle_dict.get(cid).clone_to_virtual()
        self._root = CIPORoot(
            self._ego_id,
            self._main_id,
            self._map,
            virtual_vehicle_dict,
            lon_levels,
            lat_levels,
        )
        self._probability = 1

    def generate_root_from_leaf(
            self, leaf, lon_levels, lat_levels
    ):
        """
        从CIPO Observer生成Root节点
        """

        self._root = CIPORoot(
            self._ego_id,
            self._main_id,
            self._map,
            leaf._virtual_vehicle_dict,
            lon_levels,
            lat_levels,
        )
        self._probability = leaf._risk

    def generate_root_from_partial(
        self, real_vehicle_dict, close_vehicle_id_list, lon_levels, lat_levels
    ):
        virtual_vehicle_dict = {}
        for cid in close_vehicle_id_list:
            virtual_vehicle_dict[cid] = real_vehicle_dict.get(cid).clone_to_virtual()
        self._root = CIPORoot(
            self._ego_id,
            self._main_id,
            self._map,
            virtual_vehicle_dict,
            lon_levels,
            lat_levels,
        )

    def grow_tree(self):
        """
        由根节点生成叶子结点
        返回叶子节点列表与两类叶子节点的数量
        """
        if self._root == None:
            raise ValueError("You have not generate a root node.")

        lon_leaves, num_lon = self._root.generate_leaves("longitude")
        lat_leaves, num_lat = self._root.generate_leaves("lateral")
        self._leaves = lon_leaves + lat_leaves
        self._num_lon_leaves = num_lon
        self._num_lat_leaves = num_lat

        return self._leaves, num_lon, num_lat

    def grow_tree_full(self):
        """
        由根节点生成全部叶子结点
        返回叶子节点列表与两类叶子节点的数量
        """
        if self._root == None:
            raise ValueError("You have not generate a root node.")

        leaves = self._root.generate_leaves_full()
        self._leaves = leaves

        return self._leaves


