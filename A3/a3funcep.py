# Ms Pacman Agent
from ale_py import ALEInterface, roms
import random
from collections import defaultdict
import numpy as np
import time
import pickle
import os

class MsPacmanAgent:
    def __init__(self, file="epweights.pkl"):
        self.ale = ALEInterface()
        self.ale.setBool("display_screen", False)
        self.ale.setBool("sound", False)
        self.ale.setInt("frame_skip", 4)  
        
        self.ale.loadROM("MSPACMAN.BIN")
        #self.ale.setInt("random_seed", 123)

        # prev action to detect stuck state
        self.prev_action = None

        # Initialize Q-Learning Agent variables
        self.legal_actions = self.ale.getLegalActionSet()
        self.num_actions = len(self.legal_actions)

        self.s = None
        self.a = None
        self.r = None
        self.gamma = 0.99 
        self.alpha = 0.3
        # Epsilon greedy exploration approach
        self.epsilon = 0.9  # 90% random exploration initially
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.99995  # Decay per action

        # Optimistic estimate of best possible reward
        self.R_plus = 3000.0
        self.Ne = 5.0

        # Function approximation weights and features
        self.weights = np.random.normal(0, 0.1, (10, self.num_actions))  # 8 features

        self.file = file
        self.load_weights()

        # sleep to ensure the game window is ready
        #time.sleep(1)

    def load_weights(self):
        if os.path.exists(self.file):
            try:
                with open(self.file, 'rb') as f:
                    data = pickle.load(f)

                # Load weights from file
                self.weights = data['weights']
            except Exception as e:
                print(f"Error loading weights from {self.file}: {e}")
        else:
            print(f"No existing weights found at {self.file}. Starting with random.")

    def save_weights(self):
        data = {
            'weights': self.weights
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

        # Calculate proximity to nearest ghost
        ghost_distances = []
        for gx, gy in ghost_positions:
            if gx > 0 and gy > 0:
                dist = abs(player_x - gx) + abs(player_y - gy)
                ghost_distances.append(dist)

        # Check nearest ghost distance and proximity
        if ghost_distances:
            min_distance = min(ghost_distances)
            if min_distance <= 1:
                proximity = 0  # Very close
            elif min_distance <= 5:
                proximity = 1  # Close
            elif min_distance <= 10:
                proximity = 2  # Medium
            else:
                proximity = 3  # Far
        else:
            proximity = 3  # No ghosts detected, treat as far

        direction = ram[56] // 64

        prev_player_x = getattr(self, 'prev_player_x', player_x)
        prev_player_y = getattr(self, 'prev_player_y', player_y)
        # Check if agent is stuck
        is_stuck = (player_x == prev_player_x and player_y == prev_player_y and hasattr(self, 'prev_action') and self.prev_action is not None)
        self.prev_player_x = player_x
        self.prev_player_y = player_y
        if is_stuck:
            stuck_indicator = 1
        else:
            stuck_indicator = 0

        dots_eaten_count = ram[119] // 16
        player_score = ram[120] 
        num_lives = ram[123]

        return (player_x, player_y, proximity, direction, dots_eaten_count, stuck_indicator, player_score, num_lives)

    def is_terminal(self, state):
        return self.ale.game_over()
    
    def extract_features(self, state):
        player_x, player_y, proximity, direction, dots_eaten_count, stuck_indicator, player_score, num_lives = state
        features = np.zeros(10)

        # Noramalized positions
        features[0] = player_x / 15.0
        features[1] = player_y / 15.0
        features[2] = (player_x / 15.0) * (player_y / 15.0) # Positions interaction
        features[3] = proximity / 3.0
        features[4] = direction / 3.0
        features[5] = stuck_indicator
        features[6] = dots_eaten_count / 15.0
        features[7] = player_score / 1000.0
        features[8] = num_lives / 3.0
        features[9] = 1.0  # Bias term

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
        
        
    def exploration_function(self, q_values):
        # Epsilon-greedy: with probability epsilon, pick random action
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(q_values)

    def q_learning_agent(self, percept, learning=True):
        s_prime, r_prime = percept

        # Update weights only if learning
        if self.s is not None and learning:
            self.update_weights(self.s, self.a, self.r, s_prime, self.is_terminal(s_prime))

        # find argmax_a' f(Q[s',a'], Nsa[s',a'])    
        if not self.is_terminal(s_prime):
            q_values = self.get_all_qvalues(s_prime)
            best_action = self.exploration_function(q_values)

            self.prev_action = best_action

            # Update agent state
            self.s = s_prime
            self.a = best_action
            self.r = r_prime
            return best_action
        else:
            self.prev_action = None
            # Terminal state
            self.s = s_prime
            self.a = None
            self.r = r_prime
            return None

    def run(self, num_episodes=100, learning=True):
        episode_rewards = []

        for episode in range(num_episodes):
            initial_state = self.reset()
            reward = 0
            total_reward = 0
            current_state = initial_state
            steps = 0
            previous_lives = self.ale.lives()

            while not self.is_terminal(current_state) and steps < 1000:
                # Generate action from Q-learning agent
                percept = (current_state, reward)
                a = self.q_learning_agent(percept, learning=learning)
                if a is None:
                    break  

                action = self.legal_actions[a]

                # Add penalty for losing life or being stuck, encourage survival
                raw_reward = self.ale.act(action)
                current_lives = self.ale.lives()
                penalty = 0
                if current_lives < previous_lives:
                    penalty = -10
                    previous_lives = current_lives
                if current_state[5] == 1:
                    penalty -= 20
                
                reward = raw_reward + penalty
                current_state = self.get_state()
                total_reward += raw_reward
                steps += 1

            # Final state update
            if self.is_terminal(current_state):
                final_percept = (current_state, reward)
                self.q_learning_agent(final_percept, learning=learning)
            episode_rewards.append(total_reward)

            # Decay epsilon after each episode
            if learning:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Update weights
            if learning and (episode + 1) % 100 == 0:
                self.save_weights()

            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode > 0 else total_reward
                print(f"Episode {episode}: Reward = {total_reward}, Avg Last 10 = {avg_reward:.2f}, Epsilon = {self.epsilon:.3f} for a3weightedep.py")

        if learning:
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

