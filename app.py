from factory.util import get_default_factory, get_small_default_factory, factory_string, draw_box, draw_boxes
from factory.agents import RandomAgent
# from factory.environments import FactoryEnv

import time
import os
import streamlit as st
import cv2
import numpy as np

CONVERT_FROM_BGR = False


def main():
    readme_text = st.markdown(get_file_content_as_string(os.path.expanduser("README.md")))

    st.sidebar.title("Factory Solver Settings")
    app_mode = st.sidebar.selectbox("Choose the app mode", [
        "Show instructions",
        "Run the app",
        "Show models source code",
        "Show controls source code",
        "Show agents source code",
        "Show environments source code"
    ])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show models source code":
        readme_text.empty()
        st.code(get_file_content_as_string("factory/models.py"))
    elif app_mode == "Show controls source code":
        readme_text.empty()
        st.code(get_file_content_as_string("factory/controls.py"))
    elif app_mode == "Show agents source code":
        readme_text.empty()
        st.code(get_file_content_as_string("factory/agents.py"))
    elif app_mode == "Show environments source code":
        readme_text.empty()
        st.code(get_file_content_as_string("factory/environments.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()


def run_the_app():
    st.sidebar.markdown("# Factory parameters")

    model = st.sidebar.selectbox("Choose a Factory model:", [
        "Default Factory",
        "Small Factory"
    ])

    seed = st.sidebar.slider("Random seed", 1000, 2000, 1337, 1)
    num_tables = st.sidebar.slider("Number of tables", 1, 12, 7, 1)
    num_cores = st.sidebar.slider("Number of cores", 1, num_tables, 1, 1)
    num_phases = st.sidebar.slider("Number of core phases", 1, 8, 1, 1)

    if model == "Default Factory":
        factory = get_default_factory(seed, num_tables, num_cores, num_phases)
        img_name = "./assets/large_factory.jpg"
    else:
        factory = get_small_default_factory(seed, num_tables, num_cores, num_phases)
        img_name = "./assets/small_factory.jpg"
    st.sidebar.markdown("# Agents")

    multi_agent = st.sidebar.checkbox("Multi-Agent")

    agent_type = st.sidebar.selectbox("Choose an agent type:", [
        "Random agent",
        "Simple heuristic agent",
        "RLlib agent"
    ])

    table_agent = st.sidebar.selectbox("Choose an agent to control (single-agent):", [
        f"TableAgent_{i}" for i in range(num_tables)
    ])
    table_idx = int(table_agent[11])

    if agent_type == "Random agent":
        agent = RandomAgent(factory.tables[table_idx], factory, table_agent)
    else:
        raise NotImplementedError

    st.sidebar.markdown("# Environment")

    env_type = st.sidebar.selectbox("Choose an environment:", [
        "FactoryEnv (gym)",
        "FactoryVectorEnv (RLlib)",
        "FactoryMultiAgentEnv (RLlib)"
    ])
    if env_type == "FactoryEnv (gym)":
        pass
        # env = FactoryEnv(factory, 5)
    else:
        raise NotImplementedError

    st.markdown("# Simulation")

    speed = st.slider("Simulation speed", 1, 250, 2, 1)
    steps = st.slider("Maximum number of steps", 100, 1000, 250, 1)

    start = st.button('Start Simulation')
    top_text = st.empty()
    factory_img = st.empty()

    original_image = load_image(img_name)

    if multi_agent:
        multi_agent = [RandomAgent(t, factory) for t in factory.tables]

    if start:
        for table_idx in range(steps):
            if multi_agent:
                agent = multi_agent[table_idx % len(multi_agent)]
            action = agent.compute_action()
            top_text.empty()
            top_text.text("Agent: " + str(agent.get_location().name) +
                          " | Location: " + str(agent.get_location().coordinates) +
                          "\nIntended action: " + str(action) + "\nResult: " + str(agent.take_action(action)))
            factory_img.empty()
            image = draw_boxes(factory, original_image)
            factory_img.image(image.astype(np.uint8), use_column_width=True)
            time.sleep(1.0 / speed)
            if factory.done():
                break
        top_text.empty()
        remaining_cores = len([t for t in factory.tables if t.has_core()])
        top_text.text(f"Simulation completed, {num_cores - remaining_cores} of total {num_cores} delivered.")


def load_image(file_name="./large_factory.jpg"):
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    return image


@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    return open(path).read()


if __name__ == "__main__":
    main()
