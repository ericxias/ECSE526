# Ms Pacman Agent
from ale_py import ALEInterface, roms
import random
from collections import defaultdict
import numpy as np
import time
import pickle
import os

class MsPacmanAgent:
    def __init__(self, file="pacmanqtable.pkl"):
        self.ale = ALEInterface()
        self.ale.setBool("display_screen", False)
        self.ale.setBool("sound", False)
        self.ale.setInt("frame_skip", 4)  
        
        self.ale.loadROM("MSPACMAN.BIN")
        #self.ale.setInt("random_seed", 123)

        # Initialize Q-Learning Agent variables
        self.legal_actions = self.ale.getLegalActionSet()
        self.num_actions = len(self.legal_actions)
        self.Q = defaultdict(lambda: defaultdict(float))
        self.Nsa = defaultdict(lambda: defaultdict(int))
        self.s = None
        self.a = None
        self.r = None
        self.gamma = 0.99  
        self.alpha = 0.2
        self.prev_action = None
        # Optimistic estimate of best possible reward
        self.R_plus = 3000.0
        self.Ne = 5.0

        self.file = file
        self.load_qtable()

        # sleep to ensure the game window is ready
        time.sleep(1)

    def load_qtable(self):
        if os.path.exists(self.file):
            try:
                with open(self.file, 'rb') as f:
                    data = pickle.load(f)

                self.Q = defaultdict(lambda: defaultdict(float))
                self.Nsa = defaultdict(lambda: defaultdict(int))

                # Load Q-table from file
                for state, actions in data['Q'].items():
                    for action, value in actions.items():
                        self.Q[state][action] = value

                for state, actions in data['Nsa'].items():
                    for action, value in actions.items():
                        self.Nsa[state][action] = value

            except Exception as e:
                print(f"Error loading Q-table from {self.file}: {e}")
        else:
            print(f"No existing Q-table found at {self.file}. Starting fresh.")                

    def save_qtable(self):
        data = {
            'Q': dict(self.Q),
            'Nsa': dict(self.Nsa)
        }

        with open(self.file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Q-table saved to {self.file}")


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

        min_distance = 0
        # Check nearest ghost distance and proximity
        if ghost_distances:
            min_distance = min(ghost_distances)

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

        dots_eaten_count = ram[119] / 151.0
        fruit_x = ram[11] // 16
        fruit_y = ram[17] // 16
        player_score = ram[120] 
        num_lives = ram[123]

        return (player_x, player_y, min_distance, direction, dots_eaten_count, stuck_indicator, fruit_x, fruit_y, player_score, num_lives)

    def is_terminal(self, state):
        return self.ale.game_over()

    def exploration_function(self, q_value, nsa):
        if nsa < self.Ne:
            return self.R_plus
        return q_value

    # Exploratory Q-Learning Agent, written utilizing pseudocode from AIMA textbook
    def q_learning_agent(self, percept, learning=True):
        s_prime, r_prime = percept

        # if terminal state, then Q[s, None] <- r'
        if self.s is not None and learning:
            if self.is_terminal(self.s):
                self.Q[self.s][None] = r_prime
            # if s is not None increment Nsa[s,a]
            self.Nsa[self.s][self.a] += 1

            if not self.is_terminal(s_prime):
                # Compute max_a' Q[s',a']
                max_q = float('-inf')
                for a in range(self.num_actions):
                    q_value = self.Q[s_prime][a]
                    if q_value > max_q:
                        max_q = q_value
            else:
                max_q = 0.0
            
            # Q[s,a] <- Q[s,a] + alpha(Nsa[s,a]) × (r + gamma × max_a' Q[s',a'] - Q[s,a])
            q_current = self.Q[self.s][self.a]
            self.Q[self.s][self.a] = q_current + self.alpha * (self.r + self.gamma * max_q - q_current)
        
        if not self.is_terminal(s_prime):
            best_action = None
            best_value = float('-inf')

            # find argmax_a' f(Q[s',a'], Nsa[s',a'])
            for a in range(self.num_actions):
                if learning:
                    exploration = self.exploration_function(self.Q[s_prime][a], self.Nsa[s_prime][a])
                else:
                    exploration = self.Q[s_prime][a]

                if exploration > best_value:
                    best_value = exploration
                    best_action = a
            
            # Update agent state
            self.prev_action = best_action
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

            while not self.is_terminal(current_state) and steps < 1000:
                # Generate action from Q-learning agent with percept
                percept = (current_state, reward)
                a = self.q_learning_agent(percept)
                #print(f"Current State: {current_state}, Action: {a}, Reward: {reward}")
                if a is None:
                    break  

                action = self.legal_actions[a]
                
                reward = self.ale.act(action)
                current_state = self.get_state()
                total_reward += reward
                steps += 1

            
            # Final state update
            if self.is_terminal(current_state):
                final_percept = (current_state, reward)
                self.q_learning_agent(final_percept)
            episode_rewards.append(total_reward)

             # Update Q-table
            if (episode + 1) % 100 == 0:
                self.save_qtable()

            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode > 0 else total_reward
                print(f"Episode {episode}: Reward = {total_reward}, Avg Last 10 = {avg_reward:.2f} for a3opt.py")
                print(f"  Q-table size: {len(self.Q)}, Total state-action pairs: {sum(len(actions) for actions in self.Q.values())}")
        if learning:
            self.save_qtable()
        return episode_rewards



    def reset(self):
        self.ale.reset_game()
        self.s = None
        self.a = None
        self.r = None
        return self.get_state()
    
def main():
    agent = MsPacmanAgent()
    rewards = agent.run(num_episodes=1, learning=True)
    print("Episode rewards:", rewards)

if __name__ == "__main__":
    main()

