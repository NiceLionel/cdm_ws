import carla
import time
import csv
import torch
import pygame
import utils.globalvalues as gv
import utils.extendmath as emath
import decisionmodels.CDM.cognitivedrivermodel as cdm
import decisionmodels.CDM.observer as obs
import decisionmodels.CDM.risk as rsk
import decisionmodels.CDM.reward as rwd
import decisionmodels.CDM.decision as dcs
import dstructures.enumeratetree as tree
import decisionmodels.IDM.idmcontroller as idm
import decisionmodels.D2RL.d2rldecisionmodel as d2rl
import sensors.rgbcamera as rgbc
from monitors.posmatrix import PositionMatrix
from monitors.monitor import FiniteStateMachine
from vehicles.realvehicle import RealVehicle
from model.mask_lstm import NaiveLSTM


def initialize():
    """
    全局初始化
    """
    print("Initializing...")

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

    # 获取蓝图库
    blueprint_library = world.get_blueprint_library()

    # 选择一个车辆蓝图
    vehicle_bp = blueprint_library.filter("model3")[0]  # 所有车辆蓝图都为特斯拉Model 3

    # 设置观察视角
    spectator = world.get_spectator()

    # gpu or cpu
    device = "cuda" if torch.cuda.is_available else "cpu"

    # 神经网络参数载入
    pred_net = NaiveLSTM(
        gv.INPUT_SIZE, gv.HIDDEN_SIZE, gv.NUM_CLASSES, gv.NUM_LAYERS
    ).to(device)
    pred_net.load_state_dict(torch.load("model/Traj_2lanes.pt"))

    print("Initialize done.")

    return (
        world,
        carla_map,
        vehicle_bp,
        spectator,
        pred_net,
    )


def set_cdm_vehicle(
    vehicle, driving_style, main_id, map, spawn_point, init_velocity, is_main=False
):
    """
    生成一个RealVehicle并绑定CDM
    """
    if is_main:
        id = "0"
    else:
        id = vehicle.id
    observer = obs.CIPO_Observer(id, main_id, map)
    # enumtree = tree.EnumerateTree(id, main_id, map)
    enumtrees = [tree.EnumerateTree(id, main_id, map) for _ in range(gv.NUM_OF_TREE)]
    risk_cal = rsk.CDMRisk(id, main_id, map)
    reward_cal = rwd.CDMReward(id, main_id, map)
    decision = dcs.LaplaceDecision()
    controller = cdm.NormalCognitiveDriverModel(
        driving_style, observer, enumtrees, risk_cal, reward_cal, decision
    )
    return RealVehicle(map, vehicle, spawn_point, controller, init_velocity)


def set_d2rl_vehicle(vehicle, main_id, map, spawn_point, init_velocity):
    """
    生成一个RealVehicle并绑定D2RL
    """
    id = vehicle.id
    controller = d2rl.D2RLDecisionModel(map, id, main_id)
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


def set_idm_vehicle(vehicle, main_id, map, spawn_point, init_velocity, is_main=False):
    """
    生成一个RealVehicle并绑定IDM
    """
    if is_main:
        id = "0"
    else:
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
        if abs(rel_distance) <= gv.CAR_LENGTH:
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


def run_d2rl_experiment():
    """
    该实验中所有车辆均为D2RL车辆
    """
    # 初始化
    (
        world,
        carla_map,
        vehicle_bp,
        spectator,
        pred_net,
    ) = initialize()
    # 主车生成点
    main_spawn_point = world.get_map().get_spawn_points()[264]
    # 通过读取配置文件实现车辆自动化生成
    scene_path = "scenelibrary/select_scene.csv"
    # scene_path = "scenelibrary/select_scene_cdm.csv"
    with open(scene_path, "r") as f:
        scene_list = list(csv.reader(f))
    row_idx = 0
    num_cars = 0
    while row_idx < len(scene_list):
        if len(scene_list[row_idx]) == 1:
            # 主车那行数据较少无需读取数据生成
            num_cars = int(scene_list[row_idx][0])
            # 全局车辆字典（静态）
            full_realvehicle_dict = {}
            adv_realvehicle_list = []
            # 处理主车的生成
            row_idx += 1
            scene_data = scene_list[row_idx]
            assert len(scene_data) == 3, "Main vehicle data error."
            # (init_vel, driving_style(没有则置0), 控制器("CDM", "ACDM", "OTHER"))
            scene_data = [int(scene_data[i]) for i in range(len(scene_data) - 1)] + [
                scene_data[-1]
            ]
            init_vel, driving_style, control_mode = scene_data
            # 生成车辆
            vehicle_bp.set_attribute("color", "0, 0, 0")
            main_vehicle = world.spawn_actor(vehicle_bp, main_spawn_point)
            main_id = "0"
            print("black:", main_id)
            # D2RL全局控制器
            global_controller = d2rl.D2RLGlobalDecisionModel(carla_map, main_id)
            # 配置控制器
            if control_mode == "CDM":
                full_realvehicle_dict[main_id] = set_cdm_vehicle(
                    main_vehicle,
                    driving_style,
                    main_id,
                    carla_map,
                    main_spawn_point,
                    init_vel,
                    True,
                )
            elif control_mode == "IDM":
                full_realvehicle_dict[main_id] = set_idm_vehicle(
                    main_vehicle, main_id, carla_map, main_spawn_point, init_vel, True
                )
            else:
                # 暂时还没有实现其他被测车辆
                raise ValueError("其他车辆还未实现, 主车请使用CDM。")
            for i in range(num_cars - 1):
                row_idx += 1
                scene_data = scene_list[row_idx]
                assert len(scene_data) == 5, "Main vehicle data error."
                scene_data = [
                    int(scene_data[i]) for i in range(len(scene_data) - 1)
                ] + [scene_data[-1]]
                # (init_vel, lane_id, rel_pos, driving_style(没有则置0), 控制器("CDM", "ACDM", "OTHER"))
                init_vel, lane_id, rel_pos, driving_style, control_mode = scene_data
                # 生成出生点
                spawn_point = get_spawn_point(
                    carla_map.get_waypoint(main_spawn_point.location), lane_id, rel_pos
                )
                # 生成车辆
                vehicle_bp.set_attribute("color", gv.COLOR_DICT[gv.COLOR_LIST[i]])
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                print(gv.COLOR_LIST[i], vehicle.id)
                # 配置控制器
                full_realvehicle_dict[vehicle.id] = set_d2rl_vehicle(
                    vehicle,
                    main_id,
                    carla_map,
                    spawn_point,
                    init_vel,
                )
            # 仿真步长
            step = -5

            # 终止指令
            stop = False

            # 终止后场景继续运行的时间: step
            stop_lifetime = 100

            # 假设系统运行中不会再额外生成车辆, 车辆数量与车辆字典
            full_realvehicle_id_list = list(full_realvehicle_dict.keys())

            # Monitor
            pos_matrix = PositionMatrix(main_id, carla_map)
            monitor = FiniteStateMachine(0, pos_matrix, 3)

            # 变道保持计数
            lc_maintain_count_dict = dict()
            bv_pdf_dict, bv_action_idx_dict = dict(), dict()
            for vid in full_realvehicle_id_list:
                lc_maintain_count_dict[vid] = 0
                bv_pdf_dict[vid] = None
                bv_action_idx_dict[vid] = None

            # 里程计数器
            num_round = 0
            total_round = 2
            main_s = last_main_s = -1e9
            num_unsafe = 0
            # 模拟循环
            while True:
                if step >= 0:
                    # 正常运行
                    if stop == False:
                        # 主车决策
                        if step % (gv.DECISION_DT / gv.STEP_DT) == 0:
                            full_realvehicle_dict[main_id].run_step(
                                full_realvehicle_dict
                            )
                            num_unsafe += monitor.update(full_realvehicle_dict, step)
                        if step % (gv.D2RL_DECISION_DT / gv.STEP_DT) == 0:
                            close_vehicle_id_list = full_realvehicle_dict[
                                main_id
                            ]._controller._observer.get_close_vehicle_id_list(
                                full_realvehicle_dict
                            )
                            last_bv_action_idx_dict = bv_action_idx_dict.copy()
                            bv_pdf_dict, bv_action_idx_dict = (
                                global_controller.global_decision(
                                    full_realvehicle_dict, close_vehicle_id_list, step
                                )
                            )
                            # 查看是否有变道决策
                            for key in bv_action_idx_dict:
                                if last_bv_action_idx_dict[key] in [
                                    0,
                                    1,
                                ] and lc_maintain_count_dict[key] < int(
                                    gv.LANE_CHANGE_TIME / gv.D2RL_DECISION_DT
                                ):
                                    bv_action_idx_dict[key] = last_bv_action_idx_dict[
                                        key
                                    ]
                                    lc_maintain_count_dict[key] += 1
                                else:
                                    lc_maintain_count_dict[key] = 0
                            for key in full_realvehicle_dict.keys():
                                if key != main_id:
                                    car = full_realvehicle_dict.get(key)
                                    car._control_action = (
                                        car._controller.single_decision(
                                            bv_pdf_dict, bv_action_idx_dict
                                        )
                                    )
                                    if car._scalar_velocity <= 60 / 3.6:
                                        car._control_action = (
                                            60 / 3.6 - car._scalar_velocity
                                        )
                            # if not state_hashmap.get(state):
                            #     state_hashmap[state] = 1
                            # else:
                            #     state_hashmap[state] += 1
                            # full_realvehicle_dict.get(main_id)._control_action = (
                            #     "MAINTAIN"
                            # )
                        # 场景运行时间
                        last_main_s = main_s
                        main_wp = carla_map.get_waypoint(main_vehicle.get_location())
                        main_s = emath.cal_total_in_round_length(main_wp)
                        if last_main_s > main_s:
                            num_round += 1
                            if num_round >= total_round:
                                stop = True
                                cur_step = step
                                print(
                                    "Task Finished!",
                                    f"用时{int(step * gv.STEP_DT)}秒",
                                    f"平均速度为{round(total_round * gv.ROUND_LENGTH/ max(1,int(step * gv.STEP_DT)),2)}",
                                    f"发生了{num_unsafe}次危险事件",
                                )
                        # if step > 5000:
                        #     stop = True
                        #     cur_step = step
                        # 控制车辆
                        for car in full_realvehicle_dict.values():
                            car.descrete_control()
                        # 移动观察者视角
                        spectator_tranform = carla.Transform(
                            main_vehicle.get_location() + carla.Location(z=120),
                            carla.Rotation(pitch=-90, yaw=180),
                        )
                        spectator.set_transform(spectator_tranform)
                        # 碰撞检测
                        for i in range(num_cars):
                            for j in range(i + 1, num_cars):
                                if full_realvehicle_dict[
                                    full_realvehicle_id_list[i]
                                ].collision_callback(
                                    full_realvehicle_dict[
                                        full_realvehicle_id_list[j]
                                    ]._vehicle
                                ):
                                    print(
                                        "Collide!",
                                        f"场景总共持续{int(step * gv.STEP_DT)}秒",
                                        f"任务完成度为{round((main_s + num_round * gv.ROUND_LENGTH) / (gv.ROUND_LENGTH * total_round), 2)}",
                                        f"平均速度为{round((num_round * gv.ROUND_LENGTH + main_s)/ max(1,int(step * gv.STEP_DT)),2)}m/s",
                                        f"发生了{num_unsafe}次危险事件",
                                        f"等效全程事件发生次数为{int(num_unsafe / ((main_s + num_round * gv.ROUND_LENGTH) / (gv.ROUND_LENGTH * total_round)))}",
                                    )
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
                            num_cars = 0
                            break
                    # 通知服务器进行一次模拟迭代
                    world.tick()
                    step += 1
                else:
                    world.tick()
                    step += 1
            row_idx += 1
        else:
            row_idx += 1


def run_experiment():
    # 初始化
    (
        world,
        carla_map,
        vehicle_bp,
        spectator,
        pred_net,
    ) = initialize()
    # 主车生成点
    main_spawn_point = world.get_map().get_spawn_points()[264]
    # 通过读取配置文件实现车辆自动化生成
    scene_path = "scenelibrary/select_scene.csv"
    # scene_path = "scenelibrary/select_scene_cdm.csv"
    with open(scene_path, "r") as f:
        scene_list = list(csv.reader(f))
    row_idx = 0
    num_cars = 0
    while row_idx < len(scene_list):
        if len(scene_list[row_idx]) == 1:
            # 主车那行数据较少无需读取数据生成
            num_cars = int(scene_list[row_idx][0])
            # 全局车辆字典（静态）
            full_realvehicle_dict = {}
            adv_realvehicle_list = []
            # 处理主车的生成
            row_idx += 1
            scene_data = scene_list[row_idx]
            assert len(scene_data) == 3, "Main vehicle data error."
            # (init_vel, driving_style(没有则置0), 控制器("CDM", "ACDM", "OTHER"))
            scene_data = [int(scene_data[i]) for i in range(len(scene_data) - 1)] + [
                scene_data[-1]
            ]
            init_vel, driving_style, control_mode = scene_data
            # 生成车辆
            vehicle_bp.set_attribute("color", "0, 0, 0")
            main_vehicle = world.spawn_actor(vehicle_bp, main_spawn_point)
            main_id = main_vehicle.id
            print("black:", main_id)
            # 配置传感器 (设置pygame窗口size,image_x为192的整数倍，用其他分辨率会闪屏，可能是显卡解析原因导致)
            pygame_size = {"image_x": 1152, "image_y": 600}
            cameras = rgbc.cameraManage(
                world, main_vehicle, pygame_size
            ).camaraGenarate()
            # 为渲染实例化对象
            renderObject = rgbc.RenderObject(
                pygame_size.get("image_x"), pygame_size.get("image_y")
            )
            # 采集carla世界中camera的图像
            # cameras.get("Front").listen(
            #     lambda image: rgbc.pygame_callback(image, "Front", renderObject)
            # )
            # cameras.get("Rear").listen(
            #     lambda image: rgbc.pygame_callback(image, "Rear", renderObject)
            # )
            # cameras.get("Left").listen(
            #     lambda image: rgbc.pygame_callback(image, "Left", renderObject)
            # )
            # cameras.get("Right").listen(
            #     lambda image: rgbc.pygame_callback(image, "Right", renderObject)
            # )

            # 初始化pygame显示
            # pygame.init()
            # gameDisplay = pygame.display.set_mode(
            #     (pygame_size.get("image_x"), pygame_size.get("image_y")),
            #     pygame.HWSURFACE | pygame.DOUBLEBUF,
            # )
            # 配置控制器
            if control_mode == "CDM":
                full_realvehicle_dict[main_id] = set_cdm_vehicle(
                    main_vehicle,
                    driving_style,
                    main_id,
                    carla_map,
                    main_spawn_point,
                    init_vel,
                )
            elif control_mode == "IDM":
                full_realvehicle_dict[main_id] = set_idm_vehicle(
                    main_vehicle, main_id, carla_map, main_spawn_point, init_vel
                )
            else:
                # 暂时还没有实现其他被测车辆
                raise ValueError("其他车辆还未实现, 主车请使用CDM。")
            for i in range(num_cars - 1):
                row_idx += 1
                scene_data = scene_list[row_idx]
                assert len(scene_data) == 5, "Main vehicle data error."
                scene_data = [
                    int(scene_data[i]) for i in range(len(scene_data) - 1)
                ] + [scene_data[-1]]
                # (init_vel, lane_id, rel_pos, driving_style(没有则置0), 控制器("CDM", "ACDM", "OTHER"))
                init_vel, lane_id, rel_pos, driving_style, control_mode = scene_data
                # 生成出生点
                spawn_point = get_spawn_point(
                    carla_map.get_waypoint(main_spawn_point.location), lane_id, rel_pos
                )
                # 生成车辆
                vehicle_bp.set_attribute("color", gv.COLOR_DICT[gv.COLOR_LIST[i]])
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                print(gv.COLOR_LIST[i], vehicle.id)
                # 配置控制器
                if control_mode == "CDM":
                    full_realvehicle_dict[vehicle.id] = set_cdm_vehicle(
                        vehicle,
                        driving_style,
                        main_id,
                        carla_map,
                        spawn_point,
                        init_vel,
                    )
                if control_mode == "ACDM":
                    full_realvehicle_dict[vehicle.id] = set_acdm_vehicle(
                        vehicle,
                        driving_style,
                        main_id,
                        carla_map,
                        spawn_point,
                        init_vel,
                    )
                    adv_realvehicle_list.append(vehicle.id)
            # 仿真步长
            step = -5

            # 终止指令
            stop = False

            # 终止后场景继续运行的时间: step
            stop_lifetime = 100

            # 假设系统运行中不会再额外生成车辆, 车辆数量与车辆字典
            full_realvehicle_id_list = list(full_realvehicle_dict.keys())

            # Monitor
            pos_matrix = PositionMatrix(main_id, carla_map)
            monitor = FiniteStateMachine(0, pos_matrix, 3)

            # 里程计数器
            num_round = 0
            total_round = 2
            main_s = last_main_s = -1e9
            num_unsafe = 0
            # 模拟循环
            world.tick()
            while True:
                if step >= 0:
                    # 正常运行
                    if stop == False:
                        # CDM决策
                        if step % (gv.DECISION_DT / gv.STEP_DT) == 0:
                            for car in full_realvehicle_dict.values():
                                if car._vehicle.id in adv_realvehicle_list:
                                    car.run_step(full_realvehicle_dict, pred_net)
                                else:
                                    car.run_step(full_realvehicle_dict)
                            num_unsafe += monitor.update(full_realvehicle_dict, step)
                            # if not state_hashmap.get(state):
                            #     state_hashmap[state] = 1
                            # else:
                            #     state_hashmap[state] += 1
                            # full_realvehicle_dict.get(main_id)._control_action = (
                            #     "MAINTAIN"
                            # )
                        # 场景运行时间
                        last_main_s = main_s
                        main_wp = carla_map.get_waypoint(main_vehicle.get_location())
                        main_s = emath.cal_total_in_round_length(main_wp)
                        if last_main_s > main_s:
                            num_round += 1
                            if num_round >= total_round:
                                stop = True
                                cur_step = step
                                print(
                                    "Task Finished!",
                                    f"用时{int(step * gv.STEP_DT)}秒",
                                    f"平均速度为{round(total_round * gv.ROUND_LENGTH/ max(1,int(step * gv.STEP_DT)),2)}",
                                    f"发生了{num_unsafe}次危险事件",
                                )
                        # if step > 5000:
                        #     stop = True
                        #     cur_step = step
                        # 控制车辆
                        for car in full_realvehicle_dict.values():
                            car.descrete_control()
                        # 按帧更新渲染的 Camera 画面
                        # gameDisplay.blit(renderObject.surface, (0, 0))
                        # pygame.display.flip()
                        # 移动观察者视角
                        spectator_tranform = carla.Transform(
                            main_vehicle.get_location() + carla.Location(z=120),
                            carla.Rotation(pitch=-90, yaw=180),
                        )
                        spectator.set_transform(spectator_tranform)
                        # 碰撞检测
                        for i in range(num_cars):
                            for j in range(i + 1, num_cars):
                                if full_realvehicle_dict[
                                    full_realvehicle_id_list[i]
                                ].collision_callback(
                                    full_realvehicle_dict[
                                        full_realvehicle_id_list[j]
                                    ]._vehicle
                                ):
                                    print(
                                        "Collide!",
                                        f"场景总共持续{int(step * gv.STEP_DT)}秒",
                                        f"任务完成度为{round((main_s + num_round * gv.ROUND_LENGTH) / (gv.ROUND_LENGTH * total_round), 2)}",
                                        f"平均速度为{round((num_round * gv.ROUND_LENGTH + main_s)/ max(1,int(step * gv.STEP_DT)),2)}m/s",
                                        f"发生了{num_unsafe}次危险事件",
                                        f"等效全程事件发生次数为{int(num_unsafe / ((main_s + num_round * gv.ROUND_LENGTH) / (gv.ROUND_LENGTH * total_round)))}",
                                    )
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
                            num_cars = 0
                            camera = cameras.values()
                            for cam in camera:
                                cam.stop
                            # pygame.quit()
                            break
                    # 通知服务器进行一次模拟迭代
                    world.tick()
                    step += 1
                else:
                    world.tick()
                    step += 1
            row_idx += 1
        else:
            row_idx += 1


if __name__ == "__main__":
    run_experiment()
    # run_d2rl_experiment()
