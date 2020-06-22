from factory.util.rollout import run
import ray.rllib.agents.ppo as ppo
import ray
import os

ray.init(
    webui_host='127.0.0.1',log_to_driver=True,
    memory=10000 * 1024 * 1024,
    local_mode=True
)

# e.g. "./PPO/PPO_factory_0_num_sgd_iter=10,sgd_minibatch_size=128,train_batch_size=8000_2020-06-19_15-27-57ru1fqejg/checkpoint_91/checkpoint-91"
checkpoint = os.environ.get("CHECKPOINT_FILE")
run(checkpoint=checkpoint, cls=ppo.PPOTrainer, steps=1500)
