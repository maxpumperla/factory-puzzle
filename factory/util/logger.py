from ray.tune.logger import Logger
from factory.config import SIMULATION_CONFIG


class DeepKitLogger(Logger):
    """DeepKit logger."""

    def _init(self):
        import deepkit
        account = SIMULATION_CONFIG.get("deepkit_account")
        self.experiment = deepkit.experiment(new=True)
        self.experiment.set_full_config(self.config)

    def on_result(self, result):
        for key, value in result.items():
            if not isinstance(value, float):
                continue
            if "episode" in key:
                self.experiment.log_metric(key, value)
