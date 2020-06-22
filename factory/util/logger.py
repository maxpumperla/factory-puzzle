from ray.tune.logger import Logger
import copy
from ..config import SIMULATION_CONFIG

class DeepKitLogger(Logger):

    def _init(self):
        import deepkit
        self.experiment = deepkit.experiment(new=True)
        config = copy.deepcopy(self.config)
        config.update(SIMULATION_CONFIG)
        # Note: this causes deepkit to crash
        if "vf_clip_param" in config.keys():
            config.pop("vf_clip_param")
        self.experiment.set_full_config(config)

    def on_result(self, result):
        for key, value in result.items():
            if not isinstance(value, float):
                continue
            if "episode" in key:
                self.experiment.log_metric(key, value)
