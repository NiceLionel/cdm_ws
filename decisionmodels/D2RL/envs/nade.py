import math
from decisionmodels.D2RL.mtlsp.controller.vehicle_controller.idmcontroller import (
    IDMController,
)
from decisionmodels.D2RL.mtlsp.controller.vehicle_controller.globalcontroller import (
    DummyGlobalController,
)
from decisionmodels.D2RL.envs.nde import *
from decisionmodels.D2RL.controller.nadecontroller import NADEBackgroundController
from decisionmodels.D2RL.controller.nadeglobalcontroller import NADEBVGlobalController
from decisionmodels.D2RL.nadeinfoextractor import NADEInfoExtractor


class NADE(NDE):
    def __init__(self, BVController=NADEBackgroundController, cav_model="RL"):
        if cav_model == "IDM":
            cav_controller = IDMController
        else:
            raise ValueError("Unknown AV controller!")
        super().__init__(
            AVController=cav_controller,
            BVController=BVController,
            AVGlobalController=DummyGlobalController,
            BVGlobalController=NADEBVGlobalController,
            info_extractor=NADEInfoExtractor,
        )
        self.initial_weight = 1

    # @profile
    def _step(self):
        """NADE subscribes all the departed vehicles and decides how to control the background vehicles."""
        super()._step()
