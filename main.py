import stormpy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
import os
from typing import List, Set
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

# Function to load Jani model using Stormpy
def load_jani_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JANI file not found: {path}")

    # Parse jani file using stormpy
    jani_program, properties = stormpy.parse_jani_model(path)

    print("=== Parsed JANI Program ===")
    print(f"Program name: {jani_program.name}")
    print(f"Number of automata: {len(jani_program.automata)}")

    print("\n=== Properties ===")
    for i, prop in enumerate(properties):
        print(f"- Property {i}: {prop.name}")
        print(f"  Formula: {prop.raw_formula}")

    # Build Stormpy Model
    model = stormpy.build_model(jani_program, properties)

    print("\n=== Model Information ===")
    print(f"Model type: {model.model_type}")
    print(f"Number of states: {model.nr_states}")
    print(f"Number of transitions: {model.nr_transitions}")

    for state in model.states:
        if (len(state.labels) == 1):
            print(state)
            print(state.labels)

    if model.nr_states > 0:
        initial_states = list(model.initial_states)
        print(f"Initial states: {initial_states}")
    else:
        raise ValueError("Model has no states")

    return model, properties
    
# Function to inspect the model structure
def inspect_model_structure(model):
    print("\n=== Model Structure Inspection ===")
    print(f"Model type: {type(model)}")
    print(f"Has state labeling: {hasattr(model, 'state_labeling')}")
    print(f"Has labeling: {hasattr(model, 'labeling')}")
    print(f"Has transition matrix: {hasattr(model, 'transition_matrix')}")
    
    if model.nr_states > 0:
        state_0 = model.states[0]
        print(f"State 0 type: {type(state_0)}")
        print(f"State 0 attributes: {[attr for attr in dir(state_0) if not attr.startswith('_')]}")
        
        tm = model.transition_matrix
        print(f"Transition matrix type: {type(tm)}")
        print(f"Transition matrix size: {tm.nr_rows} x {tm.nr_columns}")
        print(f"Number of entries: {tm.nr_entries}")
        
        row_group_start = model.transition_matrix.get_row_group_start(0)
        row_group_end = model.transition_matrix.get_row_group_end(0)
        print(f"State 0 row group: {row_group_start} to {row_group_end}")


# ---------- STEP 2: Define Gymnasium-compatible wrapper ----------
class JANIStormEnv(gym.Env):
    def __init__(self, model, properties=None, target_label: str = "target", max_steps: int = 1000):
        super().__init__()
        self.model = model
        self.properties = properties
        self.max_steps = max_steps
        self.step_count = 0
        
        # Validate model
        if model.nr_states == 0:
            raise ValueError("Model has no states")
        
        self.states = list(range(model.nr_states))
        
        # Get initial state
        initial_states = list(model.initial_states)
        if not initial_states:
            raise ValueError("Model has no initial states")
        self.initial_state = initial_states[0]
        self.current_state = self.initial_state
        
        # Extract actions and build transition information
        self._build_action_space()
        self._build_transition_dict()
        
        # Set up observation space
        self.observation_space = spaces.Discrete(len(self.states))
        
        # Find target states
        self.target_states = self._find_target_states(target_label)
        print(f"Found {len(self.target_states)} target states: {self.target_states}")
        
        # If no target states found, use a default goal (e.g., highest numbered state)
        if not self.target_states:
            print("Warning: No target states found, using highest numbered state as target")
            self.target_states = {max(self.states)}

            ###########################################################################################################################################################
            # self.target_states = {738, 741, 765, 767, 779, 788, 800, 810, 812, 817, 826, 831, 835, 838, 847, 853, 855, 859, 866, 868, 869, 871, 874, 879, 885, 892}
            self.target_states = {1033}

            print("\n\n\n\n")
            print(self.target_states)
            print("\n\n\n\n")
            ###########################################################################################################################################################



    def _build_action_space(self):
        """Build action space by examining the transition matrix and choice labeling."""
        all_actions = set()
        
        # Method 2: Try to extract from transition matrix structure
        if not all_actions and hasattr(self.model, 'transition_matrix'):
            tm = self.model.transition_matrix
            
            # For MDP models, we can use row groups to identify actions
            if hasattr(tm, 'get_row_group_start') and hasattr(tm, 'get_row_group_end'):
                try:
                    for state_id in range(min(10, self.model.nr_states)):  # Check first 10 states
                        row_start = tm.get_row_group_start(state_id)
                        row_end = tm.get_row_group_end(state_id)
                        num_actions = row_end - row_start
                        for action_idx in range(num_actions):
                            all_actions.add(action_idx)
                except Exception as e:
                    print(f"Could not extract actions from row groups: {e}")
        
        # Method 3: Default fallback - assume numbered actions
        if not all_actions:
            print("Warning: No actions found, using default numbered actions")
            # Try to estimate number of actions from transition matrix structure
            if hasattr(self.model, 'transition_matrix'):
                tm = self.model.transition_matrix
                if hasattr(tm, 'get_row_group_start') and hasattr(tm, 'get_row_group_end'):
                    max_actions = 0
                    for state_id in range(min(self.model.nr_states, 100)):
                        try:
                            row_start = tm.get_row_group_start(state_id)
                            row_end = tm.get_row_group_end(state_id)
                            num_actions = row_end - row_start
                            max_actions = max(max_actions, num_actions)
                        except:
                            continue
                    if max_actions > 0:
                        all_actions = set(range(max_actions))
                    else:
                        all_actions = {0}  # Single action fallback
                else:
                    all_actions = {0}  # Single action fallback
            else:
                all_actions = {0}  # Single action fallback
        
        self.actions = sorted(list(all_actions))
        self.action_space = spaces.Discrete(len(self.actions))
        
        # Create action mappings
        self.action_to_index = {action: i for i, action in enumerate(self.actions)}
        self.index_to_action = {i: action for action, i in self.action_to_index.items()}
        
        print(f"Actions found: {self.actions}")
        print(f"Action space size: {len(self.actions)}")

    def _build_transition_dict(self):
        """Build transition dictionary using the transition matrix."""
        self.transition_dict = {}
        
        if not hasattr(self.model, 'transition_matrix'):
            print("Warning: No transition matrix found")
            return
        
        tm = self.model.transition_matrix
        
        for state_id in range(self.model.nr_states):
            self.transition_dict[state_id] = {}
            
            try:
                # Get the row group for this state (contains all actions)
                if hasattr(tm, 'get_row_group_start') and hasattr(tm, 'get_row_group_end'):
                    row_start = tm.get_row_group_start(state_id)
                    row_end = tm.get_row_group_end(state_id)
                    
                    # Each row in the group represents one action
                    for action_idx, row in enumerate(range(row_start, row_end)):
                        action = action_idx  # Use index as action identifier
                        
                        transitions = []
                        # Get all transitions from this row
                        for entry in tm.get_row(row):
                            target_state = entry.column
                            probability = entry.value()
                            if probability > 0:  # Only include non-zero probability transitions
                                transitions.append((target_state, probability))
                        
                        if transitions:
                            self.transition_dict[state_id][action] = transitions
                
                else:
                    # Fallback: treat each state as having one action leading to itself
                    print(f"Warning: Cannot access row groups, using fallback transitions")
                    self.transition_dict[state_id][0] = [(state_id, 1.0)]
                    
            except Exception as e:
                print(f"Error building transitions for state {state_id}: {e}")
                # Fallback: self-loop
                self.transition_dict[state_id][0] = [(state_id, 1.0)]
        
        # Verify we have transitions
        total_transitions = sum(len(actions) for actions in self.transition_dict.values())
        print(f"Built transition dictionary with {total_transitions} state-action pairs")

    def _find_target_states(self, label: str) -> Set[int]:
        """Find states with the target label."""
        target_states = set()

        try:
            # Try different ways to access state labels
            labeling = None

            if hasattr(self.model, 'state_labeling'):
                labeling = self.model.state_labeling
            elif hasattr(self.model, 'labeling'):
                labeling = self.model.labeling

            if labeling is None:
                print(f"Warning: No labeling found, cannot find '{label}' states")
                return target_states
            
            print(f"Labeling type: {type(labeling)}")
            print(f"Labeling attributes: {[attr for attr in dir(labeling) if not attr.startswith('_')]}")
            
            # Try to get available labels

            available_labels = set()
            if hasattr(labeling, 'get_labels'):
                try:
                    available_labels = set(labeling.get_labels())
                    print(f"Available labels: {available_labels}")
                except:
                    print("Could not get labels list")
            
            # Try different methods to check for the target label
            if hasattr(labeling, 'contains_label'):
                if labeling.contains_label(label):
                    print(f"Found label '{label}' in model")
                    # Try to get states with this label
                    if hasattr(labeling, 'get_states'):
                        try:
                            labeled_states = labeling.get_states(label)
                            target_states.update(labeled_states)
                        except:
                            # Try alternative method
                            for state_id in range(self.model.nr_states):
                                try:
                                    if hasattr(labeling, 'has_state_label') and labeling.has_state_label(state_id, label):
                                        target_states.add(state_id)
                                    elif hasattr(labeling, 'get_labels_of_state'):
                                        state_labels = labeling.get_labels_of_state(state_id)
                                        if label in state_labels:
                                            target_states.add(state_id)
                                except:
                                    continue
                else:
                    print(f"Label '{label}' not found in model")
            else:
                # Try to check each state individually
                for state_id in range(self.model.nr_states):
                    try:
                        if hasattr(labeling, 'get_labels_of_state'):
                            state_labels = labeling.get_labels_of_state(state_id)
                            if label in state_labels:
                                target_states.add(state_id)
                        elif hasattr(labeling, 'has_state_label'):
                            if labeling.has_state_label(state_id, label):
                                target_states.add(state_id)
                    except:
                        continue
            
            # If we still haven't found the target label, try some common alternatives
            if not target_states and available_labels:
                common_target_labels = ['target', 'goal', 'accept', 'final', 'done', 'success']
                for alt_label in common_target_labels:
                    if alt_label in available_labels and alt_label != label:
                        print(f"Trying alternative label: '{alt_label}'")
                        try:
                            for state_id in range(self.model.nr_states):
                                if hasattr(labeling, 'get_labels_of_state'):
                                    state_labels = labeling.get_labels_of_state(state_id)
                                    if alt_label in state_labels:
                                        target_states.add(state_id)
                                elif hasattr(labeling, 'has_state_label'):
                                    if labeling.has_state_label(state_id, alt_label):
                                        target_states.add(state_id)
                            if target_states:
                                print(f"Found target states using label '{alt_label}': {target_states}")
                                break
                        except:
                            continue
            
        except Exception as e:
            print(f"Error finding target states: {e}")
        
        return target_states

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_state = self.initial_state
        self.step_count = 0
        return self.current_state, {}

    def step(self, action_idx: int):

        action_idx = int(action_idx)

        """Execute one step in the environment."""
        if action_idx >= len(self.actions):
            raise ValueError(f"Invalid action index: {action_idx}")

        
        action = self.index_to_action[action_idx]

        transitions = self.transition_dict.get(self.current_state, {}).get(action, [])
        
        if not transitions:
            # print("no no")
            # No valid transitions, stay in current state
            reward = -1 # Small penalty for invalid action
            terminated = False
            truncated = self.step_count >= self.max_steps
            return self.current_state, reward, terminated, truncated, {"action": action, "valid_action": False}
        
        # Sample next state based on transition probabilities
        if len(transitions) == 1:
            next_state = transitions[0][0]
        else:
            next_states, probs = zip(*transitions)
            next_state = np.random.choice(next_states, p=probs)
        
        # Calculate reward
        if next_state in self.target_states:
            print("yes")

        reward = 100.0 if next_state in self.target_states else -0.1  # Small step penalty
        
        # Check termination conditions
        terminated = next_state in self.target_states
        truncated = self.step_count >= self.max_steps
        
        self.current_state = next_state
        self.step_count += 1
        
        info = {
            "action": action,
            "valid_action": True,
            "step_count": self.step_count,
            "is_target": next_state in self.target_states
        }
        
        return next_state, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the current state."""
        target_info = " (TARGET)" if self.current_state in self.target_states else ""
        print(f"Step {self.step_count}: State {self.current_state}{target_info}")
        
        # Show available actions
        available_actions = list(self.transition_dict.get(self.current_state, {}).keys())
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

    model_path = "blocksworld.5.v1.jani"
    # model_path = "elevators.a-3-3.v1.jani"
    jani_model, properties = load_jani_model(model_path)
    
    inspect_model_structure(jani_model)
    
    print("\nCreating environment...")
    env = JANIStormEnv(jani_model, properties, target_label="target", max_steps=200)
    
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
        print(f"Action: {action}, Reward: {reward:.3f}, Done: {terminated or truncated}")
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
        lambda: JANIStormEnv(jani_model, properties, target_label="target", max_steps=3000), 
        n_envs=1
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

    policy_kwargs = dict(
        net_arch=[512, 512, 256, 256, 128]  # 5 hidden layers
    )

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
        policy_kwargs=policy_kwargs
    )
    
    print("Starting training...")
    model.learn(total_timesteps=300000)
    
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