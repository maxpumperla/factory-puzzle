"""Modified example of specifying an autoregressive action distribution.
In an action space with multiple components (e.g., Tuple(a1, a2, a3)), you might
want a2 to be sampled based on the sampled value of a1, and a3 to be sampled based on both a1 and a2, i.e.,
a2_sampled ~ P(a2 | a1_sampled, obs), a3_sampled ~ P(a3 | a1_sampled, a2_sampled, obs). Normally, a1, a2 and a3 would be sampled
independently.
To do this, you need both a custom model that implements the autoregressive
pattern, and a custom action distribution class that leverages that model.
This examples shows both.
"""

import gym
from gym.spaces import Discrete, Tuple
import argparse
import random
import numpy as np

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

from ray.tune import sample_from
from ray.tune.schedulers import PopulationBasedTraining

tf = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")  # try PG, PPO, IMPALA
parser.add_argument("--stop", type=int, default=300)
parser.add_argument("--num-cpus", type=int, default=0)

# ValueError: Expected tuple space for actions [1]: Tuple(Discrete(2), Discrete(2), Discrete(2))


class CorrelatedActionsEnv(gym.Env):
    """Simple env in which the policy has to emit a tuple of equal actions.
    The best score would be ~200 reward."""

    def __init__(self, _):
        self.observation_space = Discrete(2)
        self.action_space = Tuple([Discrete(2), Discrete(2), Discrete(2)])
        self.last_observation = None

    def reset(self):
        self.t = 0
        self.last_observation = random.choice([0, 1])
        return self.last_observation

    def step(self, action):
        self.t += 1
        a1, a2, a3 = action
        reward = 0
        if a1 == self.last:
            reward += 5
        # encourage correlation between a1 and a2
        if a2 == a1:
            reward += 5
        # encourage correlation between a1, a2, and a3
        if a3 == a1 and a3 == a2:
            reward += 5
        done = self.t > 20
        self.last = random.choice([0, 1])
        return self.last, reward, done, {}


class AutoregressiveOutputThreeTupleTwoTwoTwo(ActionDistribution):
    """Action distribution P(a1, a2, a3) = P(a1) * P(a2 | a1) * P(a3 | a1, a2)"""

    @staticmethod
    def required_model_output_shape(self, model_config):
        # hyperparameter choice
        return 256  # controls model output feature vector size. 16 used for 2-tuple case. Make larger for n-tuples with larger n?

    def deterministic_sample(self):
        # First, sample a1.
        a1_dist = self._a1_distribution()
        a1 = a1_dist.deterministic_sample()

        # Sample a2 conditioned on a1.
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.deterministic_sample()

        # Sample a3 conditioned on a2 and a1.
        a3_dist = self._a3_distribution(a1, a2)
        a3 = a3_dist.deterministic_sample()

        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2) + a3_dist.logp(a3)

        return (a1, a2, a3)


    def sample(self):
        # first, sample a1
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()

        # sample a2 conditioned on a1
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        # self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)

        # sample a3 conditioned on a1, a2
        a3_dist = self._a3_distribution(a1, a2)
        a3 = a3_dist.sample()

        self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2) + a3_dist.logp(a3)

        # return the action tuple
        return (a1, a2, a3)

    def logp(self, actions):
        a1, a2, a3 = actions[:, 0], actions[:, 1], actions[:, 2]
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        a1_logits, a2_logits, a3_logits = self.model.action_model([self.inputs, a1_vec, a2_vec]) # a2_vec term necessary?
        return (
            Categorical(a1_logits).logp(a1) + Categorical(a2_logits).logp(a2) + Categorical(a3_logits).logp(a3))

    def sampled_action_logp(self):
        return tf.exp(self._action_logp)

    def entropy(self):
        a1_dist = self._a1_distribution()
        a1 = a1_dist.sample()
        a2_dist = self._a2_distribution(a1)
        a2 = a2_dist.sample()
        a3_dist = self._a3_distribution(a1, a2)
        return a1_dist.entropy() + a2_dist.entropy() + a3_dist.entropy()

    def kl(self, other):
        a1_dist = self._a1_distribution()
        a1_terms = a1_dist.kl(other._a1_distribution())

        a1 = a1_dist.sample()
        a2_dist = self._a2_distribution(a1)
        a2_terms = a2_dist.kl(other._a2_distribution(a1))

        a2 = a2_dist.sample()
        a3_dist = self._a3_distribution(a1, a2)
        a3_terms = a3_dist.kl(other._a3_distribution(a1, a2))
        return a1_terms + a2_terms + a3_terms

    def _a1_distribution(self):
        BATCH = tf.shape(self.inputs)[0]
        a1_logits, _, _ = self.model.action_model(
            [self.inputs, tf.zeros((BATCH, 1)), tf.zeros((BATCH, 1))])
        a1_dist = Categorical(a1_logits)
        return a1_dist

    def _a2_distribution(self, a1):
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        BATCH = tf.shape(self.inputs)[0]
        _, a2_logits, _ = self.model.action_model([self.inputs, a1_vec, tf.zeros((BATCH, 1))])
        a2_dist = Categorical(a2_logits)
        return a2_dist

    def _a3_distribution(self, a1, a2):
        a1_vec = tf.expand_dims(tf.cast(a1, tf.float32), 1)
        a2_vec = tf.expand_dims(tf.cast(a2, tf.float32), 1)
        _, _, a3_logits = self.model.action_model([self.inputs, a1_vec, a2_vec])
        a3_dist = Categorical(a3_logits)
        return a3_dist


class AutoregressiveActionsModelThreeTuple(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(AutoregressiveActionsModelThreeTuple, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        if action_space != Tuple([Discrete(2), Discrete(2), Discrete(2)]):
            raise ValueError(
                "This model only supports the [2, 2, 2] action space")

        # Inputs
        obs_input = tf.keras.layers.Input(
            shape=obs_space.shape, name="obs_input")
        ctx_input = tf.keras.layers.Input(
            shape=(num_outputs, ), name="ctx_input")
        a1_input = tf.keras.layers.Input(shape=(1, ), name="a1_input")
        a2_input = tf.keras.layers.Input(shape=(1, ), name="a2_input")

        # Output of the model (normally 'logits', but for an autoregressive
        # dist this is more like a context/feature layer encoding the obs)
        hidden_layer = tf.keras.layers.Dense(
            256, # hyperparameter choice
            name="hidden_1",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0))(obs_input)

        context = tf.keras.layers.Dense(
            num_outputs, # hyperparameter choice, set above
            name="hidden_2",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0))(hidden_layer)

        # V(s)
        value_out = tf.keras.layers.Dense(
            1, # value function prediction using shared layers, separate critic network would be better
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(context)

        # P(a1 | obs)
        a1_logits = tf.keras.layers.Dense(
            2, # number of possible choices for action 1
            name="a1_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(ctx_input)

        # P(a2 | a1)
        a2_context = tf.keras.layers.Concatenate(axis=1)(
            [ctx_input, a1_input])

        a2_hidden = tf.keras.layers.Dense(
            64, # hyperparameter choice
            name="a2_hidden",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0))(a2_context)
        a2_logits = tf.keras.layers.Dense(
            2, # number of possible choices for action 2
            name="a2_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(a2_hidden)

        # P(a3 | a1, a2)
        a3_context = tf.keras.layers.Concatenate(axis=1)(
              [a2_context, a2_input])

        a3_hidden = tf.keras.layers.Dense(
            64, # hyperparameter choice
            name="a3_hidden",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0))(a3_context)
        a3_logits = tf.keras.layers.Dense(
            2, # number of possible choices for action 3
            name="a3_logits",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(a3_hidden)

        # Base layers
        self.base_model = tf.keras.Model(obs_input, [context, value_out])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

        # Autoregressive action sampler
        self.action_model = tf.keras.Model([ctx_input, a1_input, a2_input],
                                           [a1_logits, a2_logits, a3_logits])
        self.action_model.summary()
        self.register_variables(self.action_model.variables)

    def forward(self, input_dict, state, seq_lens):
        context, self._value_out = self.base_model(input_dict["obs"])
        return context, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

pbt_scheduler = PopulationBasedTraining(
    time_attr = 'training_iteration',
    metric = 'episode_reward_mean',
    mode = 'max',
    perturbation_interval = 5,
    quantile_fraction = 0.25,
    resample_probability = 0.25,
    log_config = True,
    hyperparam_mutations = {
        'lambda': np.linspace(0.9, 1.0, 11).tolist(),
        'lr': np.logspace(-6, -2, 50).tolist(),
        'gamma': np.linspace(0.8, 0.9997, 9).tolist(),
        'clip_param': np.linspace(0.1, 0.5, 5).tolist(),
        'kl_coeff': np.linspace(0.1, 0.4, 4).tolist(),
        'kl_target': np.linspace(0.01, 0.03, 3).tolist(),
        'entropy_coeff': np.linspace(0, 0.07, 15).tolist(),
        'sgd_minibatch_size': [128, 256, 512],
        'num_sgd_iter': [1, 10, 20, 30],
        'train_batch_size': [4000, 6000]
    }
)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_cpus=6, num_gpus=0, memory=2500000000, object_store_memory=1200000000)
    ModelCatalog.register_custom_model("autoregressive_model_3_tuple",
                                       AutoregressiveActionsModelThreeTuple)
    ModelCatalog.register_custom_action_dist("autoreg_output_3_tuple_2_2_2",
                                             AutoregressiveOutputThreeTupleTwoTwoTwo)
    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        scheduler = pbt_scheduler,
        num_samples = 10,
        config={
            "env": CorrelatedActionsEnv,
            "gamma": 0.5,
            "num_gpus": 0,
            "model": {
                "custom_model": "autoregressive_model_3_tuple",
                "custom_action_dist": "autoreg_output_3_tuple_2_2_2"
            },
            'num_gpus': 0,
            'num_workers': 1,
            'use_gae': True,
            'vf_loss_coeff': 1.0,
            'vf_clip_param': np.inf,
            'gamma': sample_from(
                lambda spec: random.choice(np.linspace(0.8, 0.9997, 9).tolist())),
            'lambda': sample_from(
                lambda spec: random.choice(np.linspace(0.9, 1.0, 11).tolist())),
            'clip_param': sample_from(
                lambda spec: random.choice(np.linspace(0.2, 0.4, 3).tolist())),
            'kl_coeff': sample_from(
                lambda spec: random.choice(np.linspace(0.2, 0.4, 3).tolist())),
            'kl_target': sample_from(
                lambda spec: random.choice(np.linspace(0.01, 0.03, 3).tolist())),
            'entropy_coeff': sample_from(
                lambda spec: random.choice(np.linspace(0, 0.07, 15).tolist())),
            'lr': sample_from(
                lambda spec: random.choice(np.logspace(-5, -2, 40).tolist())),
            'num_sgd_iter': sample_from(
                lambda spec: random.choice([10, 20, 30])),
            'sgd_minibatch_size': sample_from(
                lambda spec: random.choice([128, 256, 512])),
            'train_batch_size': sample_from(
                lambda spec: random.choice([4000, 6000])),
            'batch_mode': 'complete_episodes',
        })
