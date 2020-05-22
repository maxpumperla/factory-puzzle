from factory.util import get_default_factory, get_small_default_factory, draw_boxes
from factory.agents import RayAgent, RandomAgent
from factory.controls import TableAndRailController, ActionResult, Action
# from factory.environments import RoundRobinFactoryEnv

# import ray
# import ray.rllib.agents.dqn as dqn
# from ray.tune.registry import register_env

import streamlit as st
import cv2
import numpy as np


def main():
    # init_ray()
    run_the_app()


# @st.cache()
# def init_ray():
#     ray.init(webui_host='127.0.0.1')


@st.cache(show_spinner=True, allow_output_mutation=True)
def get_agent_and_factory(model_str, seed, num_tables, num_cores, num_phases):
    if model_str == "Default Factory":
        factory = get_default_factory(seed, num_tables, num_cores, num_phases)
        img_name = "./assets/large_factory.jpg"
    else:
        factory = get_small_default_factory(seed, num_tables, num_cores, num_phases)
        img_name = "./assets/small_factory.jpg"

    policy_file_name = "assets/dqn-small-1-1-3/checkpoint_101/checkpoint-101"
    env_name = "RoundRobinFactoryEnv"
    # register_env(env_name, lambda _: RoundRobinFactoryEnv())
    #
    # agent_cls = dqn.DQNTrainer
    #
    # # TODO: just use tensorflow to load a saved_model, for now just mock the whole thing.
    # agent = RayAgent(table=factory.tables[0], factory=factory, env_name=env_name,
    #                  policy_file_name=policy_file_name, agent_cls=agent_cls)
    agent = RandomAgent(table=factory.tables[0], factory=factory)

    return agent, factory, img_name


def run_the_app():
    st.sidebar.markdown("# Factory parameters")

    model_str = st.sidebar.selectbox("Choose a Factory model:", [
        "Default Factory",
        "Small Factory"
    ])

    seed = st.sidebar.slider("Random seed", 1000, 2000, 1337, 1)
    num_tables = st.sidebar.slider("Number of tables", 1, 12, 7, 1)
    num_cores = st.sidebar.slider("Number of cores", 0, num_tables, 1, 1)
    num_phases = st.sidebar.slider("Number of core phases", 1, 5, 1, 1)

    agent, factory, img_name = get_agent_and_factory(model_str, seed, num_tables, num_cores, num_phases)

    st.markdown("# Model Serving")

    show_explanations = st.checkbox(label="How does it work?")
    if show_explanations:
        st.markdown("To get results from your deployed model, you need to specify the table you want to control "
                    "and tell the system more about the specific situation in the factory. "
                    "\n\nInternally we will process your input, feed it into the model and return a proposed action "
                    "that can be carried out in the actual factory."
                    "\n\nFor your convenience, we display the factory here and carry out the suggested action for "
                    "visual feedback. When using our API, our backend will not track the progress of your factory. "
                    "Instead, you're responsible for carrying out those actions in the real world and update the "
                    "factory state accordingly.")

    suggested = st.markdown("### Suggested action: TBD")
    factory_img = st.empty()
    top_text = st.empty()

    original_image = load_image(img_name)
    image = draw_boxes(factory, original_image)
    factory_img.image(image.astype(np.uint8), use_column_width=True)

    obs, agent_id = observation_settings(factory)

    compute_action = st.button('Compute action')

    controllers = [TableAndRailController(t, factory) for t in factory.tables]

    invalids = 0
    collisions = 0

    if compute_action:
        action: Action = agent.compute_action(obs)
        top_text.empty()
        result = controllers[agent_id].take_action(action)

        invalids += result == ActionResult.INVALID
        collisions += result == ActionResult.COLLISION
        top_text.text(f"Agent: {agent.get_location().name} | Location: {agent.get_location().coordinates}\n"
                      f"Illegal moves: {invalids} | Collisions: {collisions}\n"
                      f"Proposed action: {action.name}\nResult: {result.name}")
        suggested.markdown("### Suggested action:  Move " + action.name)
        factory_img.empty()
        image = draw_boxes(factory, original_image)
        factory_img.image(image.astype(np.uint8), use_column_width=True)

        # top_text.empty()
        # remaining_cores = len([t for t in factory.tables if t.has_core()])
        # top_text.text(f"Simulation completed, {num_cores - remaining_cores} of total {num_cores} delivered. \n"
        #               f"Illegal moves: {invalids} | Collisions: {collisions}\n"
        #               f"Steps taken to complete: {i}")


def load_image(file_name="./large_factory.jpg"):
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    return image


@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    return open(path).read()


def observation_settings(factory):
    tables = factory.tables
    nodes = factory.nodes

    table_agent = st.selectbox("Choose an agent to control:", [
        f"Table {i} (node {tables[i].node.name})" for i in range(len(tables))
    ])
    agent_id = int(table_agent[6])
    coordinates = list(tables[agent_id].node.coordinates)

    up_free = False
    can_up = st.checkbox(label="Can table move up?")
    if can_up:
        up_free = st.checkbox(label="Is the upward node free?")

    right_free = False
    can_right = st.checkbox(label="Can table move right?")
    if can_right:
        right_free = st.checkbox(label="Is the rightward node free?")

    down_free = False
    can_down = st.checkbox(label="Can table move down?")
    if can_down:
        down_free = st.checkbox(label="Is the downward node free?")

    left_free = False
    can_left = st.checkbox(label="Can table move left")
    if can_left:
        left_free = st.checkbox(label="Is the leftward node free?")

    has_core = st.checkbox(label="Does the table have a core?")
    if has_core:
        node = st.selectbox("Choose a target node for this phase: ", [
            f"Node {i}: {nodes[i].name}" for i in range(len(nodes))
        ])
        node_idx = int(node[5])
        target = list(nodes[node_idx].coordinates)
    else:
        target = [-1, -1]

    obs = np.asarray([agent_id] + coordinates +
                     [can_up, up_free, can_right, right_free, can_down, down_free, can_left, left_free] +
                     [has_core] + target)

    return obs, agent_id


if __name__ == "__main__":
    main()
