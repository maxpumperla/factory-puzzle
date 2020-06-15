from ray.tune.logger import Logger


class DeepKitLogger(Logger):

    def _init(self):
        import deepkit
        self.experiment = deepkit.experiment(new=True)
        self.experiment.set_full_config(self.config)

    def on_result(self, result):
        for key, value in result.items():
            if not isinstance(value, float):
                continue
            if "episode" in key:
                self.experiment.log_metric(key, value)
