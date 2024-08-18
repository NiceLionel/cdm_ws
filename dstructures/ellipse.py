import math
import carla
from utils.globalvalues import CAR_LENGTH
from utils.extendmath import cal_magnitude, cal_length, cal_rel_location_curve


class EllipseGenerator:
    """
    用来生成计算risk的辅助椭圆
    """

    def __init__(self, map, c1_location, c2_location, c) -> None:
        self._map = map
        self._c1_location = c1_location
        self._c2_location = c2_location
        self._c = c
        self._car_mindis = CAR_LENGTH
        self._disriskfunc_inner_b = 60
        self._disriskfunc_outer_b = 8
        self._disriskfunc_outer_xaxis_pos = 22

    def cal_risk_vector(self, car_location):
        """
        计算风险向量, 返回Vector2D的平方长度
        """
        a = (
            cal_length(
                cal_rel_location_curve(self._map, car_location, self._c1_location)
            )
            + cal_length(
                cal_rel_location_curve(self._map, car_location, self._c2_location)
            )
        ) / 2
        b = math.sqrt(max(a * a - self._c * self._c, 0))
        o_location = (self._c1_location + self._c2_location) / 2
        # 没有椭圆, 距离计算
        if abs(b) < 1.2 or a * a - self._c * self._c <= 0:
            ego_norm = cal_rel_location_curve(
                self._map, self._c2_location, self._c1_location
            )
            ego_norm = carla.Vector2D(ego_norm.x, ego_norm.y)
            ellipse_risk = self.disriskfunc_inner(
                max(
                    cal_length(
                        cal_rel_location_curve(
                            self._map, car_location, self._c1_location
                        )
                    ),
                    self._car_mindis,
                )
            )
            a = 0
            b = 0
        else:
            rel_location = cal_rel_location_curve(self._map, car_location, o_location)

            ego_norm = carla.Vector2D(
                rel_location.x / (a * a), rel_location.y / (b * b)
            )
            ellipse_risk = self.disriskfun_outer(b)
        if ego_norm.length() > 0:
            ego_norm = (
                ego_norm.make_unit_vector()
            )  # 不知道为什么文档里写的返回的是Vector3D, 应该写错了
        return (ego_norm * ellipse_risk).length()

    def disriskfun_outer(self, x):
        """
        在焦距外的风险计算
        """
        if x <= 0:
            return self._disriskfunc_outer_b

        outer_a = -self._disriskfunc_outer_b / (
            self._disriskfunc_outer_xaxis_pos * self._disriskfunc_outer_xaxis_pos
        )
        res = outer_a * x * x + self._disriskfunc_outer_b

        return max(0, res)

    def disriskfunc_inner(self, x):
        """
        在焦距内的风险计算
        """
        if x > 2 * self._c:
            return self._disriskfunc_outer_b
        if x < self._car_mindis:
            return self._disriskfunc_inner_b

        inner_a = self._disriskfunc_inner_b / ((self._c * 2 - self._car_mindis) ** 2)
        res = inner_a * ((x - self._c * 2) ** 2) + self._disriskfunc_outer_b
        return res
