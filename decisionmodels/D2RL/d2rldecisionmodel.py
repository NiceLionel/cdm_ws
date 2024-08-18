import sys
import os
from decisionmodels.D2RL.mtlsp.simulator import Simulator
from decisionmodels.D2RL.envs.nade import NADE
from decisionmodels.D2RL.controller.treesearchnadecontroller import (
    TreeSearchNADEBackgroundController,
)
from decisionmodels.D2RL.controller.nadeglobalcontroller import NADEBVGlobalController
from decisionmodels.D2RL.conf import conf
from functools import partial
from decisionmodels.D2RL.observesumo import NadeObserver
import time
import shutil
import decisionmodels.D2RL.utils as utils
import argparse
import numpy as np
import random
import utils.globalvalues as gv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="D2RL",
    metavar="N",
    help="simulation running mode, (NDE or D2RL)",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="2lane_400m_D2RL_testing",
    metavar="N",
    help="specify experiment name (debug by default)",
)
parser.add_argument(
    "--worker_id",
    type=int,
    default=0,
    metavar="N",
    help="specify the worker id (for multi-process purposes)",
)
args = parser.parse_args()

conf.experiment_config["mode"] = args.mode
conf.experiment_config["experiment_name"] = args.experiment_name
if args.mode == "NDE":
    conf.simulation_config["epsilon_setting"] = "fixed"  # not use d2rl-based agent
elif args.mode == "D2RL":
    conf.simulation_config["epsilon_setting"] = "drl"  # use d2rl-based agent
elif args.mode == "behavior_policy":
    conf.simulation_config["epsilon_setting"] = "fixed"  # use behavior policy

print(
    f"Using mode {conf.experiment_config['mode']}, epsilon_setting {conf.simulation_config['epsilon_setting']}"
)
# If running D2RL experiments, then load the D2RL agent
d2rl_agent_path = "./decisionmodels/D2RL/checkpoints/2lane_400m/model.pt"
if conf.simulation_config["epsilon_setting"] == "drl":
    try:
        conf.discriminator_agent = conf.load_discriminator_agent(
            checkpoint_path=d2rl_agent_path
        )
    except:
        print("Time out, shutting down")
        sys.exit(0)


def run_nade_experiment(episode, experiment_path):
    # Specify the running experiment
    env = NADE(
        BVController=TreeSearchNADEBackgroundController,
        cav_model=conf.experiment_config["AV_model"],
    )
    # Specify the sumo map file and sumo configuration file
    sumo_net_file_path = "./decisionmodels/D2RL/maps/2LaneHighway/2LaneHighway.net.xml"
    sumo_config_file_path = (
        "./decisionmodels/D2RL/maps/2LaneHighway/2LaneHighwayHighSpeed.sumocfg"
    )
    # Set up the simulator
    sim = Simulator(
        sumo_net_file_path=sumo_net_file_path,
        sumo_config_file_path=sumo_config_file_path,
        num_tries=50,
        step_size=0.1,
        action_step_size=0.1,
        lc_duration=1,
        track_cav=conf.simulation_config["gui_flag"],
        sublane_flag=True,
        gui_flag=conf.simulation_config["gui_flag"],
        # output=["fcd"],
        output=[],
        experiment_path=experiment_path,
    )
    sim.bind_env(env)
    # Begin the experiment running
    sim.run(episode)
    # Return the experiment result: if no crash happens, then return 0, else, return the collision with the relative importance sampling weight
    return env.info_extractor.weight_result


def run_experiments(run_experiment=run_nade_experiment):
    # get the total running episode number and the saving experiment path
    episode_num, experiment_path = utils.get_conf()
    # get the episode starting id (default 0, can be specified using --worker_id)
    start_num = int(args.worker_id) * episode_num
    # define the run single experiment function
    run_experiment_ = partial(run_experiment, experiment_path=experiment_path)
    # Run {episode_num} episodes
    weight_result = []
    run_experiments_number = 0
    crash_number = 0
    for i in range(start_num, start_num + episode_num):
        print("episode:", i)
        try:
            weight_tmp = run_experiment_(i)
            if args.mode == "NDE":
                if weight_tmp > 0:
                    crash_number += 1
                run_experiments_number += 1
                expected_run_experiments_number = i - start_num + 1
                if (i - start_num) % 50 == 0:
                    np.save(
                        experiment_path + "/weight" + str(args.worker_id) + ".npy",
                        np.array(
                            [
                                crash_number,
                                run_experiments_number,
                                expected_run_experiments_number,
                            ]
                        ),
                    )
            else:
                weight_result.append(weight_tmp)
                # Save the result every 50 episodes to reduce the disk usage (for higher computational efficiency)
                if (i - start_num) % 50 == 0:
                    np.save(
                        experiment_path + "/weight" + str(args.worker_id) + ".npy",
                        np.array(weight_result),
                    )
                run_experiments_number += 1
                expected_run_experiments_number = i - start_num + 1
        except Exception as e:
            print(e, f"Error happens at worker {args.worker_id} episode {i}")
            continue


def action_idx_to_action(idx):
    """
    33个action
    """
    low = -4
    high = 2
    gate = 0.2
    if idx == 0:
        return "SLIDE_LEFT"
    if idx == 1:
        return "SLIDE_RIGHT"
    return low + (idx - 2) * gate


class D2RLDecisionModel:
    def __init__(self, map, ego_id, main_id) -> None:
        self._map = map
        self._ego_id = ego_id
        self._main_id = main_id

    def single_decision(self, bv_pdf_dict, bv_action_idx_dict):
        if self._ego_id == self._main_id:
            self._ego_id == "0"
        bv_pdf = bv_pdf_dict.get(self._ego_id)
        if bv_pdf is None:
            return "MAINTAIN"
        if bv_action_idx_dict.get(self._ego_id) != None:
            return action_idx_to_action(bv_action_idx_dict.get(self._ego_id))
        else:
            random.seed(2024)
            return action_idx_to_action(
                random.choices(list(range(len(bv_pdf))), weights=bv_pdf)[0]
            )


class D2RLGlobalDecisionModel:
    def __init__(self, map, main_id) -> None:
        self._env = NADE(
            BVController=TreeSearchNADEBackgroundController,
            cav_model=conf.experiment_config["AV_model"],
        )
        self._controller = NADEBVGlobalController(self._env)
        self._map = map
        self._main_id = main_id

    def global_decision(self, full_vehicle_id_dict, close_vehicle_id_list, time_step):
        """
        全局控制决策
        """
        obs_dict = {}
        for veh_id in close_vehicle_id_list:
            vehicle = full_vehicle_id_dict.get(veh_id)
            #
            if vehicle._nade_observer == None:
                vehicle._nade_observer = NadeObserver(veh_id, self._main_id, self._map)
            # 前一次决策
            prev_lat = "central"
            if vehicle._control_action == "SLIDE_LEFT":
                prev_lat = "left"
            if vehicle._control_action == "SLIDE_RIGHT":
                prev_lat = "right"
            prev_lon = (
                gv.LON_ACC_DICT.get(vehicle._control_action)
                if type(vehicle._control_action) == str
                else vehicle._control_action
            )
            if veh_id == self._main_id:
                obs_dict["0"] = vehicle._nade_observer.get_nade_observe(
                    full_vehicle_id_dict, prev_lon, prev_lat
                )
            else:
                obs_dict[veh_id] = vehicle._nade_observer.get_nade_observe(
                    full_vehicle_id_dict, prev_lon, prev_lat
                )
        (
            bv_pdf_dict,
            bv_action_idx_list,
            weight_list,
            max_vehicle_criticality,
            ndd_possi_list,
            IS_possi_list,
            controlled_bvs_id_list,
            controlled_bvs_list,
            vehicle_criticality_list,
            discriminator_input,
        ) = self._controller.select_controlled_bv_and_action_exp(
            "0", obs_dict, time_step
        )
        bv_action_idx_dict = dict(list(zip(controlled_bvs_id_list, bv_action_idx_list)))
        for key in bv_action_idx_dict:
            if bv_action_idx_dict[key] != None:
                print(bv_action_idx_dict)
                break
        return bv_pdf_dict, bv_action_idx_dict


if __name__ == "__main__":
    env = NADE(
        BVController=TreeSearchNADEBackgroundController,
        cav_model=conf.experiment_config["AV_model"],
    )
    controller = NADEBVGlobalController(env)
    CAV_obs = {
        "Ego": {
            "veh_id": "0",
            "could_drive_adjacent_lane_left": True,
            "could_drive_adjacent_lane_right": False,
            "distance": 0.0,
            "heading": 90.91931823615393,
            "lane_index": 0,
            "lateral_speed": 0.0,
            "lateral_offset": 4.440892098500626e-16,
            "prev_action": {
                "lateral": "central",
                "longitudinal": 0.1591666943357335,
            },
            "position": (642.3420202511522, 42.0),
            "position3D": (642.3420202511522, 42.0, 0.0),
            "velocity": 32.05800506474409,
            "road_id": "0to1",
            "acceleration": 0.15916669433572395,
        },
        "Lead": {
            "veh_id": "1",
            "distance": 22.924520585611845,
            "velocity": 36.01753877823528,
            "position": (670.2665408367641, 42.0),
            "heading": 90.0,
            "lane_index": 0,
            "position3D": (670.2665408367641, 42.0, 0.0),
            "acceleration": -0.1999999999999602,
        },
        "LeftLead": {
            "veh_id": "2",
            "distance": 47.74342888219917,
            "velocity": 34.793126551113474,
            "position": (695.0854491333514, 46.0),
            "heading": 90.0,
            "lane_index": 1,
            "position3D": (695.0854491333514, 46.0, 0.0),
            "acceleration": 0.0,
        },
        "RightLead": None,
        "Foll": None,
        "LeftFoll": None,
        "RightFoll": None,
    }
    Car1_obs = {
        "Ego": {
            "veh_id": "1",
            "could_drive_adjacent_lane_left": False,
            "could_drive_adjacent_lane_right": True,
            "distance": 0.0,
            "heading": 90.0,
            "lane_index": 0,
            "lateral_speed": 0.0,
            "lateral_offset": 4.440892098500626e-16,
            "prev_action": {
                "lateral": "central",
                "longitudinal": 0.1591666943357335,
            },
            "position": (642.3420202511522, 42.0),
            "position3D": (642.3420202511522, 42.0, 0.0),
            "velocity": 32.05800506474409,
            "road_id": "0to1",
            "acceleration": 0.15916669433572395,
        },
        "Lead": {
            "veh_id": "0",
            "distance": 22.924520585611845,
            "velocity": 36.01753877823528,
            "position": (670.2665408367641, 42.0),
            "heading": 90.0,
            "lane_index": 0,
            "position3D": (670.2665408367641, 42.0, 0.0),
            "acceleration": -0.1999999999999602,
        },
        "LeftLead": {
            "veh_id": "2",
            "distance": 47.74342888219917,
            "velocity": 34.793126551113474,
            "position": (695.0854491333514, 46.0),
            "heading": 90.0,
            "lane_index": 1,
            "position3D": (695.0854491333514, 46.0, 0.0),
            "acceleration": 0.0,
        },
        "RightLead": None,
        "Foll": None,
        "LeftFoll": None,
        "RightFoll": None,
    }
    Car2_obs = {
        "Ego": {
            "veh_id": "2",
            "could_drive_adjacent_lane_left": True,
            "could_drive_adjacent_lane_right": False,
            "distance": 0.0,
            "heading": 90.91931823615393,
            "lane_index": 0,
            "lateral_speed": 0.0,
            "lateral_offset": 4.440892098500626e-16,
            "prev_action": {
                "lateral": "central",
                "longitudinal": 0.1591666943357335,
            },
            "position": (642.3420202511522, 42.0),
            "position3D": (642.3420202511522, 42.0, 0.0),
            "velocity": 32.05800506474409,
            "road_id": "0to1",
            "acceleration": 0.15916669433572395,
        },
        "Lead": {
            "veh_id": "1",
            "distance": 22.924520585611845,
            "velocity": 36.01753877823528,
            "position": (670.2665408367641, 42.0),
            "heading": 90.0,
            "lane_index": 0,
            "position3D": (670.2665408367641, 42.0, 0.0),
            "acceleration": -0.1999999999999602,
        },
        "LeftLead": {
            "veh_id": "0",
            "distance": 47.74342888219917,
            "velocity": 34.793126551113474,
            "position": (695.0854491333514, 46.0),
            "heading": 90.0,
            "lane_index": 1,
            "position3D": (695.0854491333514, 46.0, 0.0),
            "acceleration": 0.0,
        },
        "RightLead": None,
        "Foll": None,
        "LeftFoll": None,
        "RightFoll": None,
    }
    obs_dict = {"0": CAV_obs, "1": Car1_obs, "2": Car2_obs}
    for elem in controller.select_controlled_bv_and_action_exp("0", obs_dict, 0):
        print("New:", elem)
    # run_experiments(run_nade_experiment)
