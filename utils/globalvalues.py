# 车辆信息(tesla model3)
CAR_LENGTH = 4.791779518127441
CAR_WIDTH = 2.163450002670288
CAR_HEIGHT = 1.4876600503921509

# 颜色字典与列表
COLOR_DICT = {"red": "255, 0, 0", "green": "0, 255, 0", "white": "255, 255, 255"}
COLOR_LIST = ["red", "green", "white"] * 10  # 避免越界与重复利用

# CDM决策频率
DECISION_DT = 1

# D2RL决策频率
D2RL_DECISION_DT = 0.1

# 场景更新频率
STEP_DT = 0.02

# 变道所需时间/s
LANE_CHANGE_TIME = 1

# 场景信息
NUM_LANE = 2
LANE_ID = {"Left": -2, "Right": -3}
MIN_SPEED = 60 / 3.6
LANE_WIDTH = 4

# 观测距离
OBSERVE_DISTANCE = 100

# 观测角度
FOV = 120

# 决策树数量
NUM_OF_TREE = 3

# Road ID and Length
ROADS_LENGTH = {
    0: [0.00, 0.00],
    37: [814.84, 0.00],
    2344: [28.99, 814.84],
    38: [300.24, 843.83],
    12: [24.93, 1144.07],
    34: [276.20, 1169.00],
    35: [21.07, 1445.20],
    2035: [29.01, 1466.27],
    36: [12.21, 1495.28],
}  # [该road的长度, 之前road的总长度]

# 一圈总长度
ROUND_LENGTH = 1507.49

# 横纵动作空间
ACTION_SPACE = ["MAINTAIN", "ACCELERATE", "DECELERATE", "SLIDE_LEFT", "SLIDE_RIGHT"]
LON_ACTIONS = ["MAINTAIN", "ACCELERATE", "DECELERATE"]
LAT_ACTIONS = ["SLIDE_LEFT", "SLIDE_RIGHT"]

# 不同动作对应的纵加速度
LON_ACC_DICT = {
    "MAINTAIN": 0,
    "ACCELERATE": 1,
    "DECELERATE": -3,
    "SLIDE_LEFT": 0,
    "SLIDE_RIGHT": 0,
}

# 动作空间对应的reward分数
ACTION_SCORE = {
    "MAINTAIN": 3,
    "ACCELERATE": 4,
    "DECELERATE": 1,
    "SLIDE_LEFT": 3,
    "SLIDE_RIGHT": 3,
}

# 神经网络参数
INPUT_SIZE = 5
HIDDEN_SIZE = 64
NUM_CLASSES = 4
NUM_LAYERS = 2

# IDM参考参数（官方）
TYPICAL_IDM_VALUE = {
    "DesiredSpeed": 120 / 3.6,
    "TimeGap": 1.0,
    "MinimumGap": 2,
    "AccelerationExp": 4,
    "Acceleration": 1.0,
    "ComfortDeceleration": -2,
    "Politness": 1,
}
