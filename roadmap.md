# Roadmap

- Run first experiments with `FactoryMultiAgentEnv`, 
using one shared policy for a population of `TableAgent`s.
    - with small factory layout
    - with large factory layout
- Since we're in Python, we can quickly experiment with many different aspects:
    - try new observation space, immediately re-run
    - try new reward function, immediately re-run
    - test many different RL algorithms available in RLlib (and potentially beyond)
- If tests are successful, go back and implement solution in AnyLogic
    - use experimental multi-agent feature of the helper to set up experiments in AnyLogic
