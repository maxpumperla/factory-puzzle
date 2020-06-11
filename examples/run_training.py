"""Start this script with "python examples/dk_trainer.py"
"""
import ray
from factory.rl import get_tune_run_config
from factory.environments import *

ray.init(webui_host='127.0.0.1',log_to_driver=True, memory=10000 * 1024 * 1024)

ray.tune.registry.register_env("factory", lambda _: FactoryEnv())

trials = ray.tune.run(**get_tune_run_config())
