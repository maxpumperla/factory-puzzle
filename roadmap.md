# Roadmap

- Since we're in Python, we can quickly experiment with many different aspects:
    - try new observation space, immediately re-run
    - try new reward function, immediately re-run
    - test many different RL algorithms available in RLlib (and potentially beyond)
- If tests are successful, go back and implement solution in AnyLogic
    - use experimental multi-agent feature of the helper to set up experiments in AnyLogic


## Experiment progression

|  Factory  | Agents    |  Cores    |  Phases   | done  |
|---        |---        |---        |---        |---    |
|      small|          1|          1|          1|     no|
|      small|          1|          1|          3|     no|
|      small|          3|          1|          1|     no|
|      small|          3|          3|          1|     no|
|      large|          3|          3|          1|     no|
|      large|         10|          5|          1|     no|
|      large|         12|         10|          3|     no|
