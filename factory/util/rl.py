import ray
from ray.rllib import models
import gym
from ray import tune
from ray.tune import schedulers
import os
import random
import numpy as np
from typing import Union
from factory.config import SIMULATION_CONFIG
from copy import deepcopy


HYPER_PARAM_MUTATIONS = {
    'lambda': np.linspace(0.9, 1.0, 5).tolist(),
    'clip_param': np.linspace(0.01, 0.5, 5).tolist(),
    'entropy_coeff': np.linspace(0, 0.03, 5).tolist(),
    'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'num_sgd_iter': [5, 10, 15, 20, 30],
    'sgd_minibatch_size': [128, 256, 512, 1024, 2048],
    'train_batch_size': [4000, 6000, 8000, 10000, 12000]
}


def get_config(algo):
    result = algo.DEFAULT_CONFIG.copy()
    result["num_gpus"] = 0
    result["num_workers"] = 4
    result["eager"] = False
    result['train_batch_size'] = 8000
    result['batch_mode'] = 'complete_episodes'

    result = apply_offline_data(result)
    result = apply_masking_model(result)
    return result


def default_scheduler():
    return schedulers.PopulationBasedTraining(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        perturbation_interval=10,
        quantile_fraction=0.25,
        resample_probability=0.25,
        log_config=True,
        hyperparam_mutations=HYPER_PARAM_MUTATIONS
    )


def default_model():
    model = models.MODEL_DEFAULTS.copy()
    model['fcnet_hiddens'] = [512, 512]  # 256, 256
    return model


def get_run_config(algorithm=None,
                   local_dir=os.path.expanduser("."),
                   scheduler=default_scheduler(),
                   model=default_model()):
    if not algorithm:
        from ray.rllib.agents.dqn import DQNTrainer
        from ray.rllib.agents.ppo import PPOTrainer
        algorithm = DQNTrainer if SIMULATION_CONFIG.get("use_dqn") else PPOTrainer

    base_config = {
        'run_or_experiment': algorithm,
        'scheduler': scheduler,
        'num_samples': SIMULATION_CONFIG.get("num_samples"),
        'stop': Stopper().stop,
        'config': {
            'env': SIMULATION_CONFIG.get("env_name"),
            'num_gpus': 0,
            'num_workers': 1,
            'model': model,
            'lr': 1e-4,
            'train_batch_size': tune.sample_from(lambda spec: random.choice([4000, 8000, 12000])),
            'batch_mode': 'complete_episodes',
        },
        'local_dir': local_dir,
        'resume': False,
        'checkpoint_freq': 50,
        'checkpoint_at_end': True,
        'max_failures': 1,
        'export_formats': ['model']
    }
    if algorithm == 'PPO':
        base_config.get('config').update({
            'use_gae': True,
            'vf_loss_coeff': 1.0,
            'vf_clip_param': np.inf,
            'lambda': 0.95,
            'clip_param': 0.2,
            'entropy_coeff': 0.0,
            'num_sgd_iter': tune.sample_from(lambda spec: random.choice([10, 20, 30])),
            'sgd_minibatch_size': tune.sample_from(lambda spec: random.choice([128, 512, 2048])),
        })
    base_config["config"] = apply_masking_model(base_config.get("config"))

    return base_config


def apply_masking_model(config):
    config = deepcopy(config)
    masking = SIMULATION_CONFIG.get("masking")
    if masking:
        from ray.rllib.models import ModelCatalog
        from factory.util.wrapper import ActionMaskingTFModel, MASKING_MODEL_NAME
        ModelCatalog.register_custom_model(MASKING_MODEL_NAME, ActionMaskingTFModel)
        config['model'] = {"custom_model": MASKING_MODEL_NAME}
        if SIMULATION_CONFIG.get("use_dqn"):
            config['hiddens'] = []
            config['dueling'] = False
    return config


def apply_offline_data(config):
    config = deepcopy(config)
    if SIMULATION_CONFIG.get("use_offline_data"):
        # TODO: currently only works for DQN without masking. With masking we get
        #   ray/rllib/optimizers/replay_buffer.py", line 265, in update_priorities
        #     assert priority > 0
        #   For PPO this will fail either way with missing "advantages" key.
        #   TL;DR: generating offline data from agents seems fine, manually creating it is very difficult.
        config['input'] = "factory-offline-data"
        #     {
        #     "factory-offline-data": 0.5,
        #     "sampler": 0.5,
        # }
        config["input_evaluation"] = []
        config['explore'] = False
        config['postprocess_inputs'] = True

    return config


class Stopper:
    def __init__(self):
        # Core criteria
        self.should_stop = False  # Stop criteria met
        self.too_many_iter = False  # Max iterations
        self.too_much_time = False  # Max training time
        self.too_many_steps = False  # Max steps

        # Stopping criteria at self.early_check
        self.no_discovery_risk = False  # Value loss never changes
        self.no_converge_risk = False  # Entropy never drops

        # Convergence signals at each iteration from self.converge_check onward
        self.converged = False  # Reward mean changes very little
        # self.value_pred_adequate = False # Explained variance >= self.value_pred_threshold

        # Episode reward behaviour
        self.episode_reward_window = []
        self.episode_reward_range = 0
        self.episode_reward_mean = 0
        self.episode_reward_mean_latest = 0

        # Entropy behaviour
        self.entropy_start = 0
        self.entropy_now = 0
        self.entropy_slope = 0

        # Value loss behaviour
        self.vf_loss_window = []
        self.vf_loss_range = 0

        # Configs
        self.episode_reward_range_threshold = 0.1  # Remove with 0
        self.entropy_slope_threshold = 0.01  # Remove with -9999999
        self.vf_loss_range_threshold = 0.1  # Remove with 0

    def stop(self, _, result):
        # Core Criteria
        self.too_many_iter = result['training_iteration'] >= 250
        self.too_much_time = result['time_total_s'] >= 43200

        if not self.should_stop and (self.too_many_iter or self.too_much_time):
            self.should_stop = True
            return self.should_stop

        # Append episode rewards list used for trend behaviour
        self.episode_reward_window.append(result['episode_reward_mean'])

        # Up until early stopping filter, append value loss list to measure range
        if result['training_iteration'] <= 50:
            self.vf_loss_window.append(result.get("info").get("learner").get("default_policy").get("vf_loss"))

        # Experimental Criteria

        # Episode steps filter
        if result['training_iteration'] == 1:
            self.entropy_start = result.get("info").get("learner").get("default_policy").get("entropy") # Set start value
            # Too many steps within episode
            self.too_many_steps = result['timesteps_total'] > 200000  # Max steps
            if not self.should_stop and self.too_many_steps:
                self.should_stop = True
                return self.should_stop

        # Early stopping filter
        if result['training_iteration'] == 50:
            self.entropy_now = result.get("info").get("learner").get("default_policy").get("entropy")
            self.episode_reward_range = np.max(np.array(self.episode_reward_window)) - np.min(
                np.array(self.episode_reward_window))
            self.entropy_slope = self.entropy_now - self.entropy_start
            self.vf_loss_range = np.max(np.array(self.vf_loss_window)) - np.min(np.array(self.vf_loss_window))
            if np.abs(self.episode_reward_range) < np.abs(
                    self.episode_reward_window[0] * self.episode_reward_range_threshold):
                self.no_converge_risk = True
            elif self.entropy_slope > np.abs(self.entropy_start * self.entropy_slope_threshold):
                self.no_converge_risk = True
            if np.abs(self.vf_loss_range) < np.abs(self.vf_loss_window[0] * self.vf_loss_range_threshold):
                self.no_discovery_risk = True

            # Early stopping decision
            if not self.should_stop and (self.no_converge_risk or self.no_discovery_risk):
                self.should_stop = True
                return self.should_stop

        # Convergence Filter
        if result['training_iteration'] >= 100:
            # Episode reward range activity
            self.episode_reward_range = np.max(np.array(self.episode_reward_window[-50:])) - np.min(
                np.array(self.episode_reward_window[-50:]))
            # Episode reward mean activity
            self.episode_reward_mean = np.mean(np.array(self.episode_reward_window[-50:]))
            self.episode_reward_mean_latest = np.mean(np.array(self.episode_reward_window[-10:]))

            # Convergence check
            if (np.abs(self.episode_reward_mean_latest - self.episode_reward_mean) / np.abs(
                    self.episode_reward_mean) < self.episode_reward_range_threshold) \
                    and (np.abs(self.episode_reward_range) < np.abs(
                    np.mean(np.array(self.episode_reward_window[-50:])) * 3)):
                self.converged = True

            # Convergence stopping decision
            if not self.should_stop and self.converged:
                self.should_stop = True
                return self.should_stop

        # Returns False by default until stopping decision made
        return self.should_stop
