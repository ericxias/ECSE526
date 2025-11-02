# Ms Pacman Agent
from ale_py import ALEInterface, roms
import random
import numpy as np

class MsPacmanAgent:
    def __init__(self):
        self.ale = ALEInterface()
        self.ale.loadROM(roms.get_rom_path("MSPACMAN.bin"))
        self.ale.setInt("random_seed", 123)
        self.ale.setBool("display_screen", True)
        
        # Get legal actions
        self.legal_actions = self.ale.getLegalActionSet()
        self.num_actions = len(self.legal_actions)
        
        # MEMORY: This is how the agent remembers what it learned
        self.q_table = {}  # Q(state, action) values - regular dict
        self.visit_count = {}  # How often each state-action was tried - regular dict
        
        # Learning parameters
        self.alpha = 0.1      # Learning rate
        self.gamma = 0.99     # Discount factor  
        self.epsilon = 0.1    # Exploration rate
        
        print(f"Agent initialized with {self.num_actions} possible actions: {self.legal_actions}")
    
    def get_state(self):
        """
        Convert game screen/RAM to a state representation
        This is how the agent 'sees' the current situation
        """
        # Simple approach: use a portion of RAM
        ram = self.ale.getRAM()
        # Use positions and game status (simplified)
        pacman_x = ram[10]  # Pacman X position (example)
        pacman_y = ram[16]  # Pacman Y position (example) 
        ghost_nearby = ram[20] > 100  # Simple ghost detection (example)
        pellets_eaten = ram[50]  # Score-related (example)
        
        # Create a tuple state (hashable for dictionary keys)
        state = (pacman_x, pacman_y, ghost_nearby, pellets_eaten // 10)
        return state
    
    def get_q_value(self, state, action):
        """
        Get Q-value for state-action pair, return 0.0 if not seen before
        """
        if state not in self.q_table:
            return 0.0
        if action not in self.q_table[state]:
            return 0.0
        return self.q_table[state][action]
    
    def set_q_value(self, state, action, value):
        """
        Set Q-value for state-action pair
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = value
    
    def get_visit_count(self, state, action):
        """
        Get visit count for state-action pair, return 0 if not seen before
        """
        if state not in self.visit_count:
            return 0
        if action not in self.visit_count[state]:
            return 0
        return self.visit_count[state][action]
    
    def increment_visit_count(self, state, action):
        """
        Increment visit count for state-action pair
        """
        if state not in self.visit_count:
            self.visit_count[state] = {}
        if action not in self.visit_count[state]:
            self.visit_count[state][action] = 0
        self.visit_count[state][action] += 1
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy strategy
        This balances exploration vs exploitation
        """
        if random.random() < self.epsilon:
            # EXPLORATION: Try a random action
            action_idx = random.randint(0, self.num_actions - 1)
            print(f"  Exploring: chose random action {action_idx}")
        else:
            # EXPLOITATION: Choose the best known action
            action_idx = self.get_best_action(state)
            print(f"  Exploiting: chose best action {action_idx}")
        
        return action_idx
    
    def get_best_action(self, state):
        """
        Get the action with highest Q-value for this state
        This is how the agent uses its memory
        """
        if state not in self.q_table:
            # Never seen this state before, choose randomly
            return random.randint(0, self.num_actions - 1)
        
        # Find action with maximum Q-value
        best_action = 0
        best_value = float('-inf')
        
        for action in range(self.num_actions):
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-table based on experience
        This is how the agent learns and remembers
        """
        # Current Q-value
        current_q = self.get_q_value(state, action)
        
        # Best possible future value
        if next_state is not None:
            max_future_q = 0.0
            if next_state in self.q_table:
                # Find maximum Q-value for next state
                for a in range(self.num_actions):
                    q_val = self.get_q_value(next_state, a)
                    if q_val > max_future_q:
                        max_future_q = q_val
        else:
            max_future_q = 0  # Terminal state
        
        # Q-Learning update rule
        target = reward + self.gamma * max_future_q
        new_q = current_q + self.alpha * (target - current_q)
        
        # REMEMBER: Store the updated value
        self.set_q_value(state, action, new_q)
        self.increment_visit_count(state, action)
        
        print(f"    Learning: Q({state}, {action}) = {current_q:.3f} → {new_q:.3f} (reward={reward})")
    
    def play_episode(self):
        """
        Play one complete game episode
        This is where learning happens through experience
        """
        self.ale.reset_game()
        total_reward = 0
        step = 0
        
        # Get initial state
        current_state = self.get_state()
        print(f"\nStarting episode, initial state: {current_state}")
        
        while not self.ale.game_over() and step < 1000:  # Limit steps per episode
            # Agent decides what to do
            action_idx = self.choose_action(current_state)
            action = self.legal_actions[action_idx]
            
            # Take action in environment
            reward = self.ale.act(action)
            total_reward += reward
            
            # Observe result
            next_state = self.get_state() if not self.ale.game_over() else None
            
            # LEARN from this experience
            self.update_q_value(current_state, action_idx, reward, next_state)
            
            # Move to next state
            current_state = next_state
            step += 1
            
            if reward != 0:  # Only print when something interesting happens
                print(f"  Step {step}: Action {action_idx}, Reward {reward}, New state: {next_state}")
        
        print(f"Episode finished: {step} steps, total reward: {total_reward}")
        return total_reward
    
    def train(self, num_episodes=100):
        """
        Train the agent over multiple episodes
        """
        print(f"Training agent for {num_episodes} episodes...")
        episode_rewards = []
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1} ---")
            reward = self.play_episode()
            episode_rewards.append(reward)
            
            # Reduce exploration over time (agent becomes more confident)
            if self.epsilon > 0.01:
                self.epsilon *= 0.995
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"\nProgress: Episodes {episode-8}-{episode+1}, Average reward: {avg_reward:.2f}")
                print(f"Exploration rate (epsilon): {self.epsilon:.3f}")
                self.print_learning_stats()
        
        return episode_rewards
    
    def print_learning_stats(self):
        """
        Show what the agent has learned
        """
        total_states = len(self.q_table)
        
        # Calculate total experiences manually
        total_experiences = 0
        for state in self.visit_count:
            for action in self.visit_count[state]:
                total_experiences += self.visit_count[state][action]
        
        print(f"Learning Statistics:")
        print(f"  States encountered: {total_states}")
        print(f"  Total experiences: {total_experiences}")
        
        if total_states > 0:
            # Show some learned Q-values
            print("  Sample learned Q-values:")
            count = 0
            for state in self.q_table:
                if count >= 3:  # Only show first 3 states
                    break
                
                # Find best action for this state
                best_action = 0
                best_value = float('-inf')
                for action in self.q_table[state]:
                    if self.q_table[state][action] > best_value:
                        best_value = self.q_table[state][action]
                        best_action = action
                
                visits = self.get_visit_count(state, best_action)
                print(f"    State {state}: Best action {best_action} (Q={best_value:.3f}, visited {visits} times)")
                count += 1
    
    def test_learned_policy(self, num_episodes=5):
        """
        Test the learned policy without exploration
        """
        print(f"\nTesting learned policy (no exploration)...")
        old_epsilon = self.epsilon
        self.epsilon = 0  # No exploration during testing
        
        test_rewards = []
        for episode in range(num_episodes):
            reward = self.play_episode()
            test_rewards.append(reward)
            print(f"Test episode {episode + 1}: {reward}")
        
        self.epsilon = old_epsilon  # Restore exploration rate
        print(f"Average test reward: {np.mean(test_rewards):.2f}")
        
        return test_rewards

# Example usage
if __name__ == "__main__":
    agent = MsPacmanAgent()
    
    # Train the agent
    training_rewards = agent.train(num_episodes=50)
    
    # Test what it learned
    agent.test_learned_policy(num_episodes=5)
    
    # Show final learning summary
    print(f"\nFinal Training Summary:")
    print(f"Average reward over last 10 training episodes: {np.mean(training_rewards[-10:]):.2f}")
    agent.print_learning_stats()