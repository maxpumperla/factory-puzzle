"""Start this script with "python examples/run_training.py"
"""
import ray
from factory.rl import get_tune_run_config
from factory.environments import register_env_from_config
from factory.config import SIMULATION_CONFIG

ray.init(
    webui_host='127.0.0.1',log_to_driver=True,
    memory=10000 * 1024 * 1024,
    local_mode=SIMULATION_CONFIG.get("local_mode")
)

register_env_from_config()

trials = ray.tune.run(**get_tune_run_config())

