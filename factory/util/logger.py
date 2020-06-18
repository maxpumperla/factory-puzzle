from ray.tune.logger import Logger
import copy

class DeepKitLogger(Logger):

    def _init(self):
        import deepkit
        self.experiment = deepkit.experiment(new=True)
        # TODO: put observations here as well
        config = copy.deepcopy(self.config)
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
