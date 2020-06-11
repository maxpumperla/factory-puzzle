from factory.config import SIMULATION_CONFIG
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.agents.dqn.distributional_q_tf_model import \
    DistributionalQTFModel
from ray.rllib.utils.framework import try_import_tf
from gym.spaces import Box

tf = try_import_tf()

MASKING_MODEL_NAME = "action_masking_tf_model"
low = SIMULATION_CONFIG.get("low")
high = SIMULATION_CONFIG.get("high")
num_obs = SIMULATION_CONFIG.get("observations")
num_actions = SIMULATION_CONFIG.get("actions")

class ActionMaskingTFModel(DistributionalQTFModel):

    def __init__(self,obs_space, action_space, num_outputs,
                 model_config, name, **kw):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kw)

        self.base_model = FullyConnectedNetwork(
            Box(low, high, shape=(num_obs,)), action_space, num_actions,
            model_config, name)

        self.register_variables(self.base_model.variables())

    def forward(self, input_dict, state, seq_lens):
        logits, _ = self.base_model({
            "obs": input_dict["obs"]["observations"]
        })
        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return logits + inf_mask, state

    def value_function(self):
        return self.base_model.value_function()

    def import_from_h5(self, h5_file):
        pass
