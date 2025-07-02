import os
from typing import List, Set
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import stormpy
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env


def load_jani_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JANI file not found: {path}")

    # Parse JANI file using Stormpy
    jani_program, properties = stormpy.parse_jani_model(path)

    print("=== Parsed JANI Program ===")
    print(f"Program name: {jani_program.name}")
    print(f"Number of automata: {len(jani_program.automata)}")

    print("\n=== Properties ===")
    for i, prop in enumerate(properties):
        print(f"- Property {i}: {prop.name}")
        print(f"  Formula: {prop.raw_formula}")

    # Build the model
    model = stormpy.build_model(jani_program, properties)

    print("\n=== Model Information ===")
    print(f"Model type: {model.model_type}")
    print(f"Number of states: {model.nr_states}")
    print(f"Number of transitions: {model.nr_transitions}")

    if model.nr_states == 0:
        raise ValueError("Model has no states")

    if not model.initial_states:
        raise ValueError("Model has no initial states")

    print("\n=== States with a Single Label ===")
    for state in model.states:
        if len(state.labels) == 1:
            print(f"State: {state}")
            print(f"Label: {next(iter(state.labels))}")

    initial_states = list(model.initial_states)
    print(f"\nInitial states: {initial_states}")

    return model, properties


def inspect_model_structure(model):
    print("\n=== Model Structure Inspection ===")

    # Basic model info
    print(f"Model type: {type(model)}")
    print(f"Has state_labeling: {hasattr(model, 'state_labeling')}")
    print(f"Has labeling: {hasattr(model, 'labeling')}")
    print(f"Has transition_matrix: {hasattr(model, 'transition_matrix')}")

    # Inspect first state
    state_0 = model.states[0]
    print(f"\n--- State 0 Inspection ---")
    print(f"Type: {type(state_0)}")
    public_attrs = [attr for attr in dir(state_0) if not attr.startswith("_")]
    print(f"Attributes: {public_attrs}")

    # Inspect transition matrix
    tm = model.transition_matrix
    print(f"\n--- Transition Matrix ---")
    print(f"Type: {type(tm)}")
    print(f"Dimensions: {tm.nr_rows} x {tm.nr_columns}")
    print(f"Number of entries: {tm.nr_entries}")

    row_start = tm.get_row_group_start(0)
    row_end = tm.get_row_group_end(0)
    print(f"State 0 transitions: row group {row_start} to {row_end}")


# ---------- STEP 2: Define Gymnasium-compatible wrapper ----------
class JANIStormEnv(gym.Env):
    def __init__(self, model, properties=None, max_steps: int = 3000):
        super().__init__()
        self.model = model
        self.properties = properties
        self.max_steps = max_steps
        self.step_count = 0

        # Define state space
        self.states = list(range(model.nr_states))

        # Initialize starting state
        initial_states = list(model.initial_states)
        self.initial_state = initial_states[0]
        self.current_state = self.initial_state

        # Build action and transition representations
        self.build_action_space()
        self.build_transition_dict()

        # Define observation space
        self.observation_space = spaces.Discrete(len(self.states))

        # Identify goal states
        self.goal_states = self.get_goal_states()
        print(f"Found {len(self.goal_states)} goal states: {self.goal_states}")

    def build_action_space(self):
        """Build the action space by analyzing the transition matrix row groups."""
        all_actions = set()
        tm = self.model.transition_matrix

        # Iterate over all states in the model
        # Each state's transitions are grouped into rows corresponding to actions
        for state_id in range(self.model.nr_states):
            # Get the start and end indices of the rows for this state in the transition matrix
            row_start = tm.get_row_group_start(state_id)
            row_end = tm.get_row_group_end(state_id)

            # Number of actions is the number of rows in this group
            for action_idx in range(row_end - row_start):
                # Collect unique action indices across all states
                all_actions.add(action_idx)

        # Sort the unique actions to have a consistent order
        self.actions = sorted(all_actions)

        # Define the action space as a discrete set of these actions
        self.action_space = spaces.Discrete(len(self.actions))

        # Create mappings to convert between action values and their indices
        self.action_to_index = {action: idx for idx, action in enumerate(self.actions)}
        self.index_to_action = {
            idx: action for action, idx in self.action_to_index.items()
        }

        # Print summary information
        print(f"Actions found: {self.actions}")
        print(f"Action space size: {self.action_space.n}")

    def build_transition_dict(self):
        """Build a transition dictionary from the model's transition matrix.

        The dictionary maps each state to its possible actions,
        and each action to a list of (target_state, probability) tuples.
        """
        self.transition_dict = {}
        tm = self.model.transition_matrix

        # Iterate over all states in the model
        for state_id in range(self.model.nr_states):
            self.transition_dict[state_id] = {}

            # Get the range of rows in the transition matrix corresponding to the state
            row_start = tm.get_row_group_start(state_id)
            row_end = tm.get_row_group_end(state_id)

            # Each row in this range corresponds to a distinct action available at the state
            for local_action_idx, row in enumerate(range(row_start, row_end)):
                action = local_action_idx  # Assign action index relative to the state
                transitions = []

                # Each entry in the row represents a transition to a target state with some probability
                for entry in tm.get_row(row):
                    target_state = entry.column
                    probability = entry.value()
                    if probability > 0:
                        # Record only transitions with positive probability
                        transitions.append((target_state, probability))

                # Store transitions under the current state and action if any exist
                if transitions:
                    self.transition_dict[state_id][action] = transitions

        # Summarize the total number of state-action pairs with transitions
        total_transitions = sum(
            len(action_dict) for action_dict in self.transition_dict.values()
        )
        print(
            f"Built transition dictionary with {total_transitions} state-action pairs"
        )

    def get_goal_states(self) -> Set[int]:
        goal_states = set()

        for state in self.model.states:
            state_label = list(state.labels)
            if len(state_label) == 1:
                if state_label[0] != "init":
                    goal_states.add(state.id)

        return goal_states

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_state = self.initial_state
        self.step_count = 0
        return self.current_state, {}

    def step(self, action_idx: int):
        """Execute one step in the environment."""

        action_idx = int(action_idx)
        if action_idx >= len(self.actions):
            raise ValueError(f"Invalid action index: {action_idx}")

        action = self.index_to_action[action_idx]

        transitions = self.transition_dict.get(self.current_state, {}).get(action, [])

        if not transitions:
            # print("no no")
            # No valid transitions, stay in current state
            reward = -1  # Small penalty for invalid action
            terminated = False
            truncated = self.step_count >= self.max_steps
            return (
                self.current_state,
                reward,
                terminated,
                truncated,
                {"action": action, "valid_action": False},
            )

        # Sample next state based on transition probabilities
        if len(transitions) == 1:
            next_state = transitions[0][0]
        else:
            next_states, probs = zip(*transitions)
            next_state = np.random.choice(next_states, p=probs)

        # Calculate reward
        if next_state in self.goal_states:
            print("yes")

        reward = 100.0 if next_state in self.goal_states else -0.1  # Small step penalty

        # Check termination conditions
        terminated = next_state in self.goal_states
        truncated = self.step_count >= self.max_steps

        self.current_state = next_state
        self.step_count += 1

        info = {
            "action": action,
            "valid_action": True,
            "step_count": self.step_count,
            "is_target": next_state in self.goal_states,
        }

        return next_state, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the current state."""

        goal_info = " (Goal)" if self.current_state in self.goal_states else ""
        print(f"Step {self.step_count}: State {self.current_state}{goal_info}")

        # Show available actions
        available_actions = list(
            self.transition_dict.get(self.current_state, {}).keys()
        )
        print(f"Available actions: {available_actions}")

    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices for current state."""
        valid_actions = []
        state_actions = self.transition_dict.get(self.current_state, {})
        for action in state_actions:
            if action in self.action_to_index:
                valid_actions.append(self.action_to_index[action])
        return valid_actions


# ---------- STEP 3: Main Training Code ----------
def main():

    # model_path = "blocksworld.5.v1.jani"
    model_path = "elevators.a-3-3.v1.jani"
    jani_model, properties = load_jani_model(model_path)

    inspect_model_structure(jani_model)

    print("\nCreating environment...")
    env = JANIStormEnv(jani_model, properties, max_steps=500)

    # Test environment manually first
    print("\n=== Testing Environment ===")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    env.render()

    for i in range(5):
        valid_actions = env.get_valid_actions()
        if valid_actions:
            action = np.random.choice(valid_actions)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Action: {action}, Reward: {reward:.3f}, Done: {terminated or truncated}"
        )
        env.render()

        if terminated or truncated:
            print("Episode ended early")
            break

    # Train using DQN
    print("\n=== Training DQN Agent ===")

    # Check environment compatibility
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check warning: {e}")

    # Create vectorized environment
    vec_env = make_vec_env(
        lambda: JANIStormEnv(jani_model, properties, max_steps=3000),
        n_envs=1,
    )

    # Initialize and train DQN model
    # model = PPO(
    #     "MlpPolicy",
    #     vec_env,
    #     verbose=1,
    #     learning_rate=0.0001,
    #     # buffer_size=10000,
    #     # learning_starts=10000,
    #     # target_update_interval=5000,
    #     # train_freq=10,
    #     # exploration_fraction=0.3,
    #     # exploration_final_eps=0.1
    # )

    policy_kwargs = dict(net_arch=[512, 512, 256, 256, 128])  # 5 hidden layers

    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        # learning_rate=0.0001,
        # buffer_size=10000,
        learning_starts=10000,
        # target_update_interval=5000,
        # train_freq=10,
        # exploration_fraction=0.3,
        exploration_final_eps=0.3,
        policy_kwargs=policy_kwargs,
    )

    print("Starting training...")
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("jani_dqn_model")
    print("Model saved as 'jani_dqn_model'")

    # Test the trained agent
    print("\n=== Testing Trained Agent ===")
    obs, _ = env.reset()
    total_reward = 0

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=False)

        print(action)

        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        total_reward += reward

        if step % 10 == 0 or terminated or truncated:
            env.render()
            print(f"Step {step}, Total reward: {total_reward:.3f}")

        if terminated or truncated:
            print(f"Episode ended. Final reward: {total_reward:.3f}")
            break


if __name__ == "__main__":
    main()
