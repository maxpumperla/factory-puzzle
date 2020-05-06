from ray.tune import run
from factory.util.rl import run_config
from factory.environments import FactoryMultiAgentEnv


env = FactoryMultiAgentEnv()
trials = run(run_config(env))
