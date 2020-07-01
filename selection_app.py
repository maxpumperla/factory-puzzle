import streamlit as st
from ruamel import yaml
import subprocess
import copy

def load_schema(schema_str):
    return yaml.load(schema_str, Loader=yaml.RoundTripLoader)
    # return yaml.safe_load(schema_str)


@st.cache()
def run_training():
    cmd = "python examples/run_training.py"
    p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)
    p.wait()


def run_the_app():

    with open("deepkit.yml", 'r') as f:
        yaml_str = f.read()
    schema = copy.deepcopy(load_schema(yaml_str))

    config = generate_frontend_from_observations(schema.get("config"))
    schema["config"] = config

    show_config = st.checkbox("Show resulting config", False)
    if show_config:
        st.text(yaml.dump(config))

    st.markdown("### Save the current configuration for future runs.")

    save_config = st.button("Save configuration")
    if save_config:
        save_config_to_yaml(schema)


    st.markdown("### Start a new training run with your configuration.")
    start_training = st.button("Start training")
    if start_training:
        run_training()

    st.markdown("### Let Pathmind tune all observations and rewards for you.")
    tune = st.checkbox("Tune experiment")
    if tune:
        runs = int(st.number_input(label="Select the number of random search runs", value=10))

        start_search = st.button("Start hyper-parameter search")
        if start_search:
            original_schema = copy.deepcopy(schema)
            for i in range(runs):
                modify_schema = copy.deepcopy(original_schema)
                config = random_choice_obs_rewards(modify_schema.get("config"))
                modify_schema["config"] = config
                save_config_to_yaml(modify_schema)

                st.markdown("### Running training with the following configration.")
                st.text(yaml.dump(config))

                run_training()


def save_config_to_yaml(schema):
    with open("deepkit.yml", 'w') as f:
        f.write(yaml.dump(schema, Dumper=yaml.RoundTripDumper))


def get_obs_and_rewards(config: dict):

    observations = {k: v for k, v in config.items() if k.startswith('obs_')}
    rewards = {k: v for k, v in config.items() if k.startswith('rew_')}
    return observations, rewards


def random_choice_obs_rewards(config: dict):
    import random
    random.choice([True, False])

    observations, rewards = get_obs_and_rewards(config)

    for k, v in observations.items():
        if v:
            config[k] = random.choice([True, False])

    for k, v in rewards.items():
        if v.get("value"):
            new_val = random.choice([True, False])
            config[k] = {"value": new_val, "weight": v.get("weight")}

    return config


def generate_frontend_from_observations(config: dict):

    observations, rewards = get_obs_and_rewards(config)

    st.markdown("# Observation and reward selection")

    st.sidebar.markdown("## Observations")

    for k, v in observations.items():
        val = st.sidebar.checkbox(label=k, value=v)
        config[k] = val

    st.sidebar.markdown("## Rewards")

    for k, v in rewards.items():
        value = v.get("value")
        weight = v.get("weight")
        new_val = st.sidebar.checkbox(label=k, value=value)
        new_weight = st.sidebar.number_input(label=k + " weight", value=weight)
        config[k] = {"value": new_val, "weight": new_weight}

    return config

if __name__ == "__main__":
    run_the_app()
