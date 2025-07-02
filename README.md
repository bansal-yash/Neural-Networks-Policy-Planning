# Neural-Networks-Policy-Planning
Neural network and reinforcement learning models for efficient decision-making on classical planning benchmarks.

This project provides a custom Gymnasium environment for **model-based reinforcement learning** on **JANI models** using **Stormpy** and **Stable-Baselines3 (SB3)**. It allows parsing JANI models, building transition systems, defining goal-oriented RL environments, and training agents (e.g., DQN) to solve them.

> 🧪 **This project was developed as part of a research internship** at the *Foundations of Artificial Intelligence* group, **Saarland University**, under the supervision of **Prof. Joerg Hoffmann**.

## 📌 Features

* ✅ **JANI Parser** using Stormpy
* ✅ **Transition extraction** and state/action modeling
* ✅ **Custom Gym environment** compatible with Stable-Baselines3
* ✅ **DQN agent training and evaluation**
* ✅ Works with **probabilistic models** and labeled states

## 📂 Project Structure

```
.
├── main.py                    # Main script containing environment, model loader, and training logic
├── elevators.a-3-3.v1.jani      # Example JANI model (replaceable)
└── README.md                     # This file
```

## 🛠 Requirements

* Python 3.12
* Stormpy
* Gymnasium
* Stable-Baselines3
* NumPy
* z3

Install dependencies:

```bash
pip install stable-baselines3[extra] gymnasium numpy z3-solver
# Stormpy needs to be built from source with z3 enabled or installed from your package manager
```

## 🚀 Getting Started

### 1. Add a JANI model

Replace the `model_path` in the script with your JANI file:

```python
model_path = "elevators.a-3-3.v1.jani"
```

### 2. Run the script

```bash
python main.py
```

This will:
* Load and inspect the model
* Create the environment
* Test it with random actions
* Train a DQN agent
* Save the trained model
* Evaluate the agent

## 🎮 Environment Design

* **State space**: Discrete states from the JANI model
* **Action space**: Actions derived from Stormpy's transition matrix
* **Rewards**:
  * +100 for reaching a goal state
  * -0.1 per step to encourage efficiency
  * -1 for invalid transitions
* **Termination**: Episode ends if a goal state is reached or `max_steps` is exceeded

## 🧪 Example Output

```
Actions found: [0, 1, 2, 3]
Built transition dictionary with 378 state-action pairs
Found 15 goal states: {23, 57, ...}

Step 0: State 0
Available actions: [0, 1]
Action: 1, Reward: -0.1, Done: False
...
Episode ended. Final reward: 100.0
```

## 📈 Training Configuration

DQN is configured with:
* MLP with 5 hidden layers `[512, 512, 256, 256, 128]`
* Learning rate: `1e-4`
* Exploration: ε-greedy with final ε = 0.3
* Training steps: `100,000`

## 💾 Model Saving

Trained models are saved automatically:

```bash
jani_dqn_model.zip
```

You can reload using:

```python
from stable_baselines3 import DQN
model = DQN.load("jani_dqn_model")
```

## 📚 References

* [JANI Model Format](https://jani-spec.org/)
* [Stormpy Documentation](https://moves-rwth.github.io/stormpy/)
* [Gymnasium Docs](https://gymnasium.farama.org/)
* [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
