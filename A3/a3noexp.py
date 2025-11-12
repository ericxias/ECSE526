# Ms Pacman Agent
from ale_py import ALEInterface, roms
import random
from collections import defaultdict
import numpy as np
import time
import pickle
import os

class MsPacmanAgent:
    def __init__(self, file="new_weights.pkl"):
        self.ale = ALEInterface()
        self.ale.setBool("display_screen", False)
        self.ale.setBool("sound", False)
        self.ale.setInt("frame_skip", 4)  
        
        self.ale.loadROM("MSPACMAN.BIN")
        #self.ale.setInt("random_seed", 123)



        # Initialize Q-Learning Agent variables
        self.legal_actions = self.ale.getLegalActionSet()
        self.num_actions = len(self.legal_actions)

        self.s = None
        self.a = None
        self.r = None
        self.gamma = 0.99 
        self.alpha = 0.2
        self.Nsa = defaultdict(lambda: defaultdict(int))
        # Optimistic estimate of best possible reward
        self.R_plus = 3000.0
        self.Ne = 8.0

        # Function approximation weights - increased to 12 features for ghost positions
        self.weights = np.random.normal(0, 0.1, (12, self.num_actions))

        self.file = file
        self.load_weights()

        # sleep to ensure the game window is ready
        #time.sleep(1)

    def load_weights(self):
        if os.path.exists(self.file):
            try:
                with open(self.file, 'rb') as f:
                    data = pickle.load(f)

                self.weights = data['weights']
                self.Nsa = defaultdict(lambda: defaultdict(int), data.get('Nsa', {}))
            except Exception as e:
                print(f"Error loading weights from {self.file}: {e}")
        else:
            print(f"No existing weights found at {self.file}. Starting with random.")

    def save_weights(self):
        data = {
            'weights': self.weights,
            'Nsa': dict(self.Nsa)
        }

        with open(self.file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Weights saved to {self.file}")

    def get_state(self):
        ram = self.ale.getRAM()
    
        # RAM locations taken from the ALE documentation and Ms. Pacman specifics
        player_x = ram[10] // 16
        player_y = ram[16] // 16
        ghost_positions = [
            (ram[6] // 16, ram[12] // 16),   # Sue
            (ram[7] // 16, ram[13] // 16),   # Inky
            (ram[8] // 16, ram[14] // 16),   # Pinky
            (ram[9] // 16, ram[15] // 16)    # Blinky
        ]

        direction = ram[56] // 64
        return (player_x, player_y, tuple(ghost_positions), direction)

    def is_terminal(self, state):
        return self.ale.game_over()
    
    def calculate_min_ghost_distance(self, player_x, player_y, ghost_positions):
        min_distance = float('inf')
        for gx, gy in ghost_positions:
            if gx > 0 and gy > 0:  # Valid ghost position
                distance = abs(player_x - gx) + abs(player_y - gy)
                min_distance = min(min_distance, distance)
        return min_distance if min_distance != float('inf') else 10  # Default if no valid ghosts
    
    def extract_features(self, state):
        player_x, player_y, ghost_positions, direction, stuck_indicator = state
        features = np.zeros(12)

        # Normalize features
        features[0] = player_x / 15.0
        features[1] = player_y / 15.0
        
        # Direct relative ghost positions
        ghost_features = []
        for gx, gy in ghost_positions:
            if gx > 0 and gy > 0:  # Valid ghost
                rel_x = (gx - player_x) / 15.0  # Relative X (-1 to +1)
                rel_y = (gy - player_y) / 15.0  # Relative Y (-1 to +1)
                ghost_features.extend([rel_x, rel_y])
        
        # Pad with zeros if fewer than 4 ghosts
        while len(ghost_features) < 8: 
            ghost_features.extend([0.0, 0.0])
        
        # Take first 8 features (4 ghosts max)
        features[2:10] = ghost_features[:8]
        
        # Simple additional features
        features[10] = direction / 3.0
        features[11] = stuck_indicator
        
        return features
    
    def get_all_qvalues(self, state):
        features = self.extract_features(state)
        q_values = np.dot(self.weights.T, features)
        return q_values
    
    def get_qvalue(self, state, action):
        features = self.extract_features(state)
        q_value = np.dot(self.weights[:, action], features)
        return q_value
    
    # Based on equation 21.13 from AIMA textbook
    def update_weights(self, state, action, reward, next_state, game_over):
        # Qθ(s,a)
        q_value = self.get_qvalue(state, action)
        if game_over:
            target = reward
        else:
            # target = R(s) + gamma max_a' Qθ(s',a')
            next_q_values = self.get_all_qvalues(next_state)
            target = reward + self.gamma * np.max(next_q_values)

        # delta Qθ(s,a)/deltaθ    
        features = self.extract_features(state)
        self.weights[:, action] += self.alpha * (target - q_value) * features

    # Q-learning agent without exploration
    def q_learning_agent(self, percept):
        s_prime, r_prime = percept

        # Update weights based on previous state-action-reward
        if self.s is not None:
            self.update_weights(self.s, self.a, self.r, s_prime, self.is_terminal(s_prime))

        # Pure greedy action selection (no exploration)
        if not self.is_terminal(s_prime):
            q_values = self.get_all_qvalues(s_prime)
            
            # Simply pick action with highest Q-value
            best_action = np.argmax(q_values)


            self.Nsa[str(s_prime)][best_action] += 1
            
            self.s = s_prime
            self.a = best_action
            self.r = r_prime
            return best_action
        else:
            # Terminal state
            self.s = s_prime
            self.a = None
            self.r = r_prime
            return None
                
    def run(self, num_episodes=100):
        episode_rewards = []

        for episode in range(num_episodes):
            initial_state = self.reset()
            reward = 0
            total_reward = 0
            current_state = initial_state
            steps = 0
            previous_lives = self.ale.lives()

            exploration_actions = 0
            total_actions = 0

            while not self.is_terminal(current_state) and steps < 1000:
                current_exploration = 0
                for a in range(self.num_actions):
                    if self.Nsa[str(current_state)][a] < self.Ne:
                        current_exploration += 1
                exploration_actions += current_exploration
                total_actions += self.num_actions

                # Generate action from Q-learning agent with percept
                percept = (current_state, reward)
                a = self.q_learning_agent(percept)
                if a is None:
                    break  

                action = self.legal_actions[a]
                raw_reward = self.ale.act(action)
                current_lives = self.ale.lives()
                penalty = 0
                if current_lives < previous_lives:
                    penalty = -10
                    previous_lives = current_lives
                
                reward = raw_reward + penalty
                current_state = self.get_state()
                total_reward += raw_reward
                steps += 1

            # Final state update
            if self.is_terminal(current_state):
                final_percept = (current_state, reward)
                self.q_learning_agent(final_percept)
            episode_rewards.append(total_reward)

            # Update Q-table
            if (episode + 1) % 100 == 0:
                self.save_weights()

            exploration_rate = exploration_actions / total_actions if total_actions > 0 else 0
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode > 0 else total_reward
                print(f"Episode {episode}: Reward = {total_reward}, Avg Last 10 = {avg_reward:.2f}, Exploration Rate = {exploration_rate:.2f} for a3weighted.py")

        self.save_weights()
        return episode_rewards

    def reset(self):        
        self.ale.reset_game()
        self.s = None
        self.a = None
        self.r = None
        return self.get_state()
    
def main():
    agent = MsPacmanAgent()
    
    rewards = agent.run(num_episodes=30000)
    print("Episode rewards:", rewards)

if __name__ == "__main__":
    main()