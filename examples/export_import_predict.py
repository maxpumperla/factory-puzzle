import shutil
import random
import os
import numpy as np

from factory.environments import FactoryEnv
import tensorflow as tf

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env


ray.init()

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 4
config["eager"] = False


register_env("factory", lambda _: FactoryEnv())
trainer = ppo.PPOTrainer(config=config, env="factory")
result = trainer.train()

# Ray's built-in TF exporter
# trainer.get_policy().export_model(model_dir)
policy = trainer.get_policy()
model_dir = os.path.join(ray.utils.get_user_temp_dir(), "models")
policy.export_model(model_dir)

shutil.make_archive("saved_model", 'zip', model_dir)
# shutil.rmtree("./saved_model")



import tensorflow as tf
import ray
import os
import random
import numpy as np
# tf.executing_eagerly()
# tf.compat.v1.disable_eager_execution()

model_dir = os.path.join(ray.utils.get_user_temp_dir(), "models")
tf_model = tf.saved_model.load(model_dir)

arr = np.array([random.randint(0, 10) for _ in range(14)]).reshape(1, 14)

inputs = tf.convert_to_tensor(arr, dtype=tf.float32, name='observations')
infer = tf_model.signatures.get("serving_default")


# with tf.compat.v1.Session() as sess:
#     loaded_policy = tf.compat.v1.saved_model.loader.load(sess, ["serve"], export_dir=model_dir)
#
#     is_training_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("default_policy/is_training:0")
#     # observation_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("default_policy/observation:0")
#     prev_action_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("default_policy/action_1:0")
#     prev_reward_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("default_policy/prev_reward:0")
#     seq_lens_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("default_policy/seq_lens:0")
#
#     # out_action_prob = tf.compat.v1.get_default_graph().get_tensor_by_name("default_policy/Exp:0")
#     # out_actions = tf.compat.v1.get_default_graph().get_tensor_by_name("default_policy/Squeeze:0")
#     # out_behaviour_logits = tf.compat.v1.get_default_graph().get_tensor_by_name("default_policy/model/fc_out/BiasAdd:0")
#     # out_vf_preds = tf.compat.v1.get_default_graph().get_tensor_by_name("default_policy/Reshape:0")
#     #
#     # result = sess.run([out_actions, out_action_prob], {observation_tensor: arr})


is_training_tensor = tf.constant(False, dtype=tf.bool)
prev_reward_tensor = tf.constant([0], dtype=tf.float32)
prev_action_tensor = tf.constant([0], dtype=tf.int64)
seq_lens_tensor = tf.constant([0], dtype=tf.int32)

result =infer(observations=inputs, is_training=is_training_tensor, seq_lens=seq_lens_tensor,
              prev_action=prev_action_tensor, prev_reward=prev_reward_tensor)

action_tensor = result.get("actions_0")
int_action = int(action_tensor.numpy()[0])
