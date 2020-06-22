# RL Factory Solver

## Installation

```bash
git@github.com:maxpumperla/factory-puzzle.git
cd factory-puzzle
virtualenv venv && source venv/bin/activate
python setup.py develop
```

## Running experiments

Make sure you understand the conceptual basics of the project first by reading this document.
All parameters of an experiment are centrally defined in the `deepkit.yml` file. This includes

- The layout and configuration of the factory you want to work with
- The setup of the RL algorithm you want to try.
- The selection of observations and reward terms to use.

Note that you do _not_ need DeepKit to run experiments, plain Python 3.6+ is enough. You do
get a lot of extra perks by [installing it optionally](https://deepkit.ai/download), though.

The `examples` folder has a few scripts to get you started with your experiments. Most of
the time you'll want to modify your experiment configuration in `deepkit.yml` and then run
`python examples/run_training.py` to kick off an experiment equivalent to one running on
the Pathmind web application.

## Running the simulation demo

```bash
streamlit run app.py
```

## Running tests

```bash
pip install -e ".[dev]"
pytest .
```

## Terminology and problem statement

We're given a specified layout of a factory (2D, bird's eye view). The factory consists
of _nodes_ that are connected to each other. Nodes have _shuttles_ on them which can
either move or not. We say a node is static if its shuttle can't move. Otherwise the
node is part of a _rail_, which is a sequence of connected nodes which has precisely
one shuttle that can move along the rail.

Shuttles can bear _tables_. Tables can move to adjacent nodes with a shuttle if said 
shuttle doesn't already have a table on it. Tables can only enter a rail if the rail
shuttle is empty and adjacent to the table in question. All other movement types are
unrestricted, i.e. tables can always move from static to static nodes, from rail to
static nodes and freely move along a rail once on it. The rules imply that tables
can't "pass each other", which might cause obstructions.

Tables may or may not have a _core_ on them. A core moves by moving the table it resides
on. A core goes through one or several production _phases_. In each phase the core 
needs to be delivered to a _target node_. Once a core has gone through all phases it 
will be removed.

A _factory_ is a collection of nodes, shuttles, rails, tables and cores (with phases).
Solving the factory means to move the cores so that they pass all target nodes according
to their phases.

## Implementation

We aim for a clean separation of concerns through several modules.

- `models`: Models specify the basic building blocks for this project (what).
They model what objects are and how to interact with them.
- `controls`: Controls specify how objects change state (how).
- `environments`: In which environment are things happening? (where)
The environment specifies what agents can observe and how
they are rewarded for actions.
- `agents`: Agents use the controls of models to find smart ways to act (why).
We'll mainly use this abstraction for testing heuristics and loading `RLlib` agents.
- `simulation`: Sets up the complete _factory_ with all of its components (global state).
The factory object used in an environment is also used to track statistics and can be thought
of as the equivalent to AnyLogic's `Main` agent.
- `config`: Manages the configuration for all experiments and runs.

Additionally to these core components, to solve this problem we have Reinforcement Learning
specific modules. Ideally, at this point, if you want to extend the current functionality of
this project by means of fancier algorithms or other libraries, you should only ever touch
these:

- `rl`: this module builds up the `config` object used by Ray `tune` and `rllib`.
- `features`: here you can define observations, rewards, done conditions and information that
will be picked up by the environment.

### Models

We keep the abstractions at a minimum and e.g. get by without explicit models for
shuttles, edges, networks and the like. You have access to the following interfaces.

`Node`, `Table`, `Rail`, `Core`, `Phase`, `Factory`

### Controllers

What parts can move and how is movement defined in a factory? This is part of the
modeling process and is not as simple as it may sound at first. The basic movable
objects in our implementation are `Table`s and `Rail`s. We provide three implementations
to address control:

- `TableAndRailController`: Controls behaviour of a single `Table` in a `Factory`
    and its adjacent `Rail`s. If the agent wants to move to an available rail, it can 
    actively order the respective rail shuttle. This is the default controller and
    is unlikely to change at this point.
- `TableController`: controls behaviour of a single `Table` in a `Factory`.
    The table can only enter a rail, if the shuttle is already right next
    to it (and empty).
- `RailController`: Only controls the shuttle on its `Rail`, no tables.

Using the first controller leads to a single agent population, using the latter two
likely requires two separate agent types. The controller you want to use should be
exposed via the `do_action` function.

We define the `Action` space controller-centric by specifying the main directions in
which they could potentially move, i.e `left`, `right`, `up`, `down` and `none`. Smart
agents are responsible for figuring out the right directions and avoiding illegal
movement patterns.

### Environments

We define three separate RL environments to solve this factory problem:

- `FactoryEnv`: A straight-up OpenAI `gym.Env`. If you use this environment you need
to make sure that agents take turns round-robin (not ideal).
- `RoundRobinFactoryEnv`: Structurally equivalent to the above `FactoryEnv`, this environment
tracks which agent should move next and attributes observations and rewards accordingly.
- `MultiAgentFactoryEnv`: Implements `ray`'s `MultiAgentEnv` and is useful for either
several agent types or several policies mapping to individuals in a a population.

### Agents

We provide a simple `Agent` interface to `compute_action`s, as well as `save` and
`restore` functionality for agents. Smart `Agent`s can also be `train`ed. This setup 
allows us to test several dummy implementations against properly trained RL agents. 
Current agent implementations are:

- `RandomAgent`: Takes random actions for `Table`s and `Rail`s.
- `HeuristicAgent`: Implements a simple heuristic to move tables with cores to their
target on a shortest-path trajectory, while moving tables without cores out of the
way. Useful for comparisons.
- `RayAgent`: Simply wraps `ray`'s `Trainer` implementation, which has the exact
same interface (not exactly by accident).

