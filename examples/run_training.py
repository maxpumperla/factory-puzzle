"""Start this script with "python examples/run_training.py"
"""
import ray
from factory.rl import get_tune_run_config
from factory.environments import register_env_from_config

ray.init(webui_host='127.0.0.1',log_to_driver=True, memory=10000 * 1024 * 1024)

register_env_from_config()

trials = ray.tune.run(**get_tune_run_config())

