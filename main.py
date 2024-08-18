import carla
import csv
import torch
import utils.globalvalues as gv
import utils.extendmath as emath
import decisionmodels.CDM.cognitivedrivermodel as cdm
import decisionmodels.CDM.observer as obs
import decisionmodels.CDM.risk as rsk
import decisionmodels.CDM.reward as rwd
import decisionmodels.CDM.decision as dcs
import dstructures.enumeratetree as tree
import decisionmodels.IDM.idmcontroller as idm
from monitors.posmatrix import PositionMatrix
from monitors.monitor import FiniteStateMachine
from vehicles.realvehicle import RealVehicle
from model.mask_lstm import NaiveLSTM


def initialize():
    """
    全局初始化
    """
    print("Initializing...")
    print(torch.__version__)
    print(torch.cuda.is_available())

    # 连接到Carla服务器
    client = carla.Client("localhost", 2000)

    # 超时报错
    client.set_timeout(10.0)

    # 获取世界
    world = client.get_world()

    # 获取世界设置
    settings = world.get_settings()

    # 设置同步模式
    settings.synchronous_mode = True

    # 同步模式每帧时间间隔
    settings.fixed_delta_seconds = gv.STEP_DT

    # 应用设置
    world.apply_settings(settings)

    # 获取地图
    carla_map = world.get_map()

    # 全局车辆字典（静态）
    full_realvehicle_dict = {}
    adv_realvehicle_list = []

    # 获取蓝图库
    blueprint_library = world.get_blueprint_library()

    # 选择一个车辆蓝图
    vehicle_bp = blueprint_library.filter("model3")[0]  # 所有车辆蓝图都为特斯拉Model 3

    # 设置观察视角
    spectator = world.get_spectator()

    # gpu or cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 神经网络参数载入
    pred_net = NaiveLSTM(
        gv.INPUT_SIZE, gv.HIDDEN_SIZE, gv.NUM_CLASSES, gv.NUM_LAYERS
    ).to(device)
    pred_net.load_state_dict(torch.load("model/Traj_2lanes.pt"))

    print("Initialize done.")

    return (
        world,
        carla_map,
        full_realvehicle_dict,
        adv_realvehicle_list,
        vehicle_bp,
        spectator,
        pred_net,
    )


def set_cdm_vehicle(vehicle, driving_style, main_id, map, spawn_point, init_velocity):
    """
    生成一个RealVehicle并绑定CDM
    """
    id = vehicle.id
    observer = obs.CIPO_Observer(id, main_id, map)
    enumtrees = [tree.EnumerateTree(id, main_id, map) for _ in range(gv.NUM_OF_TREE)]
    # enumtree = tree.EnumerateTree(id, main_id, map)
    risk_cal = rsk.CDMRisk(id, main_id, map)
    reward_cal = rwd.CDMReward(id, main_id, map)
    decision = dcs.LaplaceDecision()
    controller = cdm.NormalCognitiveDriverModel(
        driving_style, observer, enumtrees, risk_cal, reward_cal, decision
    )
    return RealVehicle(map, vehicle, spawn_point, controller, init_velocity)


def set_acdm_vehicle(vehicle, driving_style, main_id, map, spawn_point, init_velocity):
    """
    生成一个RealVehicle并绑定ACDM
    """
    id = vehicle.id
    observer = obs.CIPO_Observer(id, main_id, map)
    enumtrees = [tree.EnumerateTree(id, main_id, map) for _ in range(gv.NUM_OF_TREE)]
    # enumtree = tree.EnumerateTree(id, main_id, map)
    risk_cal = rsk.CDMRisk(id, main_id, map)
    reward_cal = rwd.ACDMReward(id, main_id, map)
    decision = dcs.LaplaceDecision()
    controller = cdm.AdversarialCognitiveDriverModel(
        driving_style, observer, enumtrees, risk_cal, reward_cal, decision
    )
    return RealVehicle(map, vehicle, spawn_point, controller, init_velocity)


def set_idm_vehicle(vehicle, main_id, map, spawn_point, init_velocity):
    """
    生成一个RealVehicle并绑定IDM
    """
    id = vehicle.id
    observer = idm.IDMObserver(id, main_id, map)
    controller = idm.IntelligentDriverModel(
        observer,
        gv.TYPICAL_IDM_VALUE["TimeGap"],
        gv.TYPICAL_IDM_VALUE["DesiredSpeed"],
        gv.TYPICAL_IDM_VALUE["Acceleration"],
        gv.TYPICAL_IDM_VALUE["MinimumGap"],
        gv.TYPICAL_IDM_VALUE["AccelerationExp"],
        gv.TYPICAL_IDM_VALUE["ComfortDeceleration"],
        gv.TYPICAL_IDM_VALUE["Politness"],
    )
    return RealVehicle(map, vehicle, spawn_point, controller, init_velocity)


def get_spawn_point(main_waypoint, lane, rel_distance):
    """
    以主车为中心获取车辆生成点
    Param:
    main_spawn_point -> carla.Waypoint() 主车生成点
    lane -> int 0表示左车道, 1表示右车道
    rel_distance -> float 相对距离
    Return: carla.Transform()
    """
    if lane != 0 and lane != 1:
        raise ValueError("Wrong lane id!")
    lane_id = "Left" if lane == 0 else "Right"
    spawn_wp = None
    # 生成在相同的车道上, 避免碰撞
    if gv.LANE_ID[lane_id] == main_waypoint.lane_id:
        if rel_distance <= gv.CAR_LENGTH:
            raise ValueError("The spawn point is too close!")
        spawn_wp = main_waypoint
    # 生成在不同的车道上
    else:
        if lane == 0:
            spawn_wp = main_waypoint.get_left_lane()
        if lane == 1:
            spawn_wp = main_waypoint.get_right_lane()
    # 获取相对位置的waypoint
    if rel_distance > 0:
        spawn_wp = spawn_wp.next(rel_distance)[0]
    else:
        spawn_wp = spawn_wp.previous(-rel_distance)[0]

    # 沿道路法向量增加高度
    res = spawn_wp.transform
    res.location = emath.add_height_to_waypoint(res, gv.CAR_HEIGHT / 2)
    return res


def main():
    # 初始化
    (
        world,
        carla_map,
        full_realvehicle_dict,
        adv_realvehicle_list,
        vehicle_bp,
        spectator,
        pred_net,
    ) = initialize()
    ################## 后续通过读取配置文件实现车辆自动化生成
    # 选择生成点
    spawn_point = world.get_map().get_spawn_points()[264]
    spawn_point1 = get_spawn_point(carla_map.get_waypoint(spawn_point.location), 0, -68)
    spawn_point2 = get_spawn_point(carla_map.get_waypoint(spawn_point.location), 1, 10)
    spawn_point3 = get_spawn_point(carla_map.get_waypoint(spawn_point.location), 1, 25)
    driving_style = 40
    # 生成车辆
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    main_id = vehicle.id
    print("蓝车:", vehicle.id)
    full_realvehicle_dict[vehicle.id] = set_cdm_vehicle(
        vehicle, 60, main_id, carla_map, spawn_point, 24
    )
    # full_realvehicle_dict[vehicle.id] = set_idm_vehicle(
    #     vehicle, main_id, carla_map, spawn_point, 24
    # )
    # # 生成车辆
    # vehicle_bp.set_attribute("color", "255, 0, 0")
    # vehicle2 = world.spawn_actor(vehicle_bp, spawn_point1)
    # print("红车:", vehicle2.id)
    # full_realvehicle_dict[vehicle2.id] = set_cdm_vehicle(
    #     vehicle2, 120, main_id, carla_map, spawn_point1, 28
    # )
    # adv_realvehicle_list.append(vehicle2.id)
    # 生成车辆
    vehicle_bp.set_attribute("color", "0, 255, 0")
    vehicle3 = world.spawn_actor(vehicle_bp, spawn_point2)
    print("绿车:", vehicle3.id)
    full_realvehicle_dict[vehicle3.id] = set_acdm_vehicle(
        vehicle3, 60, main_id, carla_map, spawn_point2, 24
    )
    adv_realvehicle_list.append(vehicle3.id)
    # # 生成车辆
    # vehicle_bp.set_attribute("color", "255, 255, 255")
    # vehicle4 = world.spawn_actor(vehicle_bp, spawn_point3)
    # print("白车:", vehicle4.id)
    # full_realvehicle_dict[vehicle4.id] = set_cdm_vehicle(
    #     vehicle4, 50, main_id, carla_map, spawn_point3, 25
    # )
    # adv_realvehicle_list.append(vehicle4.id)
    ###################
    # 仿真步长
    step = 0

    # 终止指令
    stop = False

    # 终止后场景继续运行的时间: step
    stop_lifetime = 500

    # 假设系统运行中不会再额外生成车辆, 车辆数量与车辆字典
    full_realvehicle_id_list = list(full_realvehicle_dict.keys())
    num_vehicles = len(full_realvehicle_id_list)

    # Monitor
    pos_matrix = PositionMatrix(main_id, carla_map)
    monitor = FiniteStateMachine(0, pos_matrix, 3)
    # 模拟循环
    while True:
        # 正常运行
        if stop == False:
            # CDM决策
            if step % (gv.DECISION_DT / gv.STEP_DT) == 0:
                for car in full_realvehicle_dict.values():
                    if car._vehicle.id in adv_realvehicle_list:
                        car.run_step(full_realvehicle_dict, pred_net)
                    else:
                        car.run_step(full_realvehicle_dict)
                monitor.update(full_realvehicle_dict, step)
                # full_realvehicle_dict.get(vehicle.id)._control_action = "MAINTAIN"
            # 场景运行时间
            if step > 5000:
                stop = True
                cur_step = step
            # 控制车辆
            for car in full_realvehicle_dict.values():
                car.descrete_control()
            # 移动观察者视角
            spectator_tranform = carla.Transform(
                vehicle.get_location() + carla.Location(z=120),
                carla.Rotation(pitch=-90, yaw=180),
            )
            spectator.set_transform(spectator_tranform)
            # 碰撞检测
            for i in range(num_vehicles):
                for j in range(i + 1, num_vehicles):
                    if full_realvehicle_dict[
                        full_realvehicle_id_list[i]
                    ].collision_callback(
                        full_realvehicle_dict[full_realvehicle_id_list[j]]._vehicle
                    ):
                        print("Collide!")
                        stop = True
                        cur_step = step
        # 终止运行
        if stop == True:
            # LifeTime结束
            if step - cur_step >= stop_lifetime:
                # 删除所有车辆, 释放内存
                for carid in full_realvehicle_id_list:
                    full_realvehicle_dict[carid]._vehicle.destroy()
                    del full_realvehicle_dict[carid]
                full_realvehicle_id_list = []
                num_vehicles = 0
                raise KeyboardInterrupt()
        # print(vehicle.get_location())
        # print(vehicle2.get_location())
        # print(vehicle3.get_location())
        # 通知服务器进行一次模拟迭代
        world.tick()
        # time.sleep(0.01)  # 等待一帧的时间
        step += 1


if __name__ == "__main__":
    main()
