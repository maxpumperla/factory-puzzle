from ray.tune import run
from factory.util.rl import run_config
from factory.environments import MultiAgentFactoryEnv
from ray.tune.registry import register_env


register_env("factory", lambda _: MultiAgentFactoryEnv())

trials = run(**run_config(env="factory"))
