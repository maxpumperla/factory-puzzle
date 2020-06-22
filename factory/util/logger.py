from ray.tune.logger import Logger
import copy
from ..config import SIMULATION_CONFIG

class DeepKitLogger(Logger):

    def _init(self):
        import deepkit
        self.experiment = deepkit.experiment(new=True)
        experiment_list = SIMULATION_CONFIG.get("experiment_list")
        if experiment_list:
            self.experiment.set_list(experiment_list)
        config = copy.deepcopy(self.config)
        config.update(SIMULATION_CONFIG)

        # Note: this causes deepkit to crash (np.inf value)
        if "vf_clip_param" in config.keys():
            config.pop("vf_clip_param")
        self.experiment.set_full_config(config)

    def on_result(self, result):
        for key, value in result.items():
            if not isinstance(value, float):
                continue
            if "episode" in key:
                self.experiment.log_metric(key, value)
