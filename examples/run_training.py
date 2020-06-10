import ray
from ray.tune import run
from ray.tune.logger import Logger, DEFAULT_LOGGERS
from factory.util.rl import get_run_config
from factory.environments import *
from ray.tune.registry import register_env

ray.init(
    webui_host='127.0.0.1',
    log_to_driver=True,
    memory=10000 * 1024 * 1024,
    # object_store_memory=5000 * 1024 * 1024,
    # driver_object_store_memory=4000 * 1024 * 1024
)


register_env("factory", lambda _: FactoryEnv())

config = get_run_config()

class DeepKitLogger(Logger):
    """DeepKit logger."""

    def _init(self):
        import deepkit
        self.experiment = deepkit.experiment(new=True, account="deepkit")
        self.experiment.set_full_config(self.config)

    def on_result(self, result):
        for key, value in result.items():
            if not isinstance(value, float):
                continue
            if "episode" in key:
                self.experiment.log_metric(key, value)


config['loggers'] = list(DEFAULT_LOGGERS) + [DeepKitLogger]

trials = run(**config)
