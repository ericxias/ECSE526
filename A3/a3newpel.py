# Ms Pacman Agent
from ale_py import ALEInterface, roms
import random
from collections import defaultdict
import numpy as np
import time
import pickle
import os

class MsPacmanAgent:
    def __init__(self, file="distanceqtable.pkl"):
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
        self.prev_action = None
        # Distance metric weights for generalization, higher weights -> more important
        self.weights = {
            'pos_x' : 2.0,
            'pos_y' : 2.0,
            'prox' : 3.0,
            'direction' : 1.0,
            'dots_eaten' : 2.0,
            'stuck' : 2.5
        }
         # Add caching for similar states
        self.similar_states_cache = {}
        self.cache_hit_threshold = 5
        self.step_counter = 0

        self.s = None
        self.a = None
        self.r = None
        self.alpha = 0.1
        self.gamma = 0.99  
        # Optimistic estimate of best possible reward
        self.R_plus = 50.0
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

        dots_eaten = ram[119] // 16

        return (player_x, player_y, proximity, direction, dots_eaten, stuck_indicator)

    def is_terminal(self, state):
        return self.ale.game_over()

    def exploration_function(self, q_value, nsa):
        if nsa < self.Ne:
            return self.R_plus
        return q_value
    
    def calculate_distance(self, state1, state2):
        pos_x1, pos_y1, prox1, dir1, dots_eaten1,stuck1 = state1
        pos_x2, pos_y2, prox2, dir2, dots_eaten2, stuck2 = state2

        # Distance calculation with weights, using generalized Euclidean distance from 21.4
        distance = 0.0
        distance += self.weights['pos_x'] * (pos_x1 - pos_x2) ** 2
        distance += self.weights['pos_y'] * (pos_y1 - pos_y2) ** 2
        distance += self.weights['prox'] * (prox1 - prox2) ** 2
        distance += self.weights['direction'] * (dir1 - dir2) ** 2
        distance += self.weights['dots_eaten'] * (dots_eaten1 - dots_eaten2) ** 2
        distance += self.weights['stuck'] * (stuck1 - stuck2) ** 2

        return np.sqrt(distance)
    
    def similar_states(self, target, max_distance=2.0, max_states=4):
        similar_states = []
        states_checked = 0
        # set max states to check to limit computation
        max_checks = min(len(self.Q), 100)

        # Similar states within max_distance
        for state in self.Q.keys():
            if state != target:
                distance = self.calculate_distance(target, state)
                if distance <= max_distance:
                    similar_weight = 1.0 / (1.0 + distance)
                    similar_states.append((similar_weight, state, distance))

                    # early termination if enough states found -> efficiency
                    if len(similar_states) >= max_states:
                        break

                states_checked += 1
                if states_checked >= max_checks:
                    break

        # Sort by similarity weight
        similar_states.sort(reverse=True, key=lambda x: x[0])
        return similar_states[:max_states]
    
    def get_cached_similar_states(self, state):
        """Cache similar states for frequently visited states"""
        state_key = str(state)
        total_visits = sum(self.Nsa[state][a] for a in range(self.num_actions))
        
        # Only cache frequently visited states
        if total_visits >= self.cache_hit_threshold:
            if state_key not in self.similar_states_cache:
                self.similar_states_cache[state_key] = self.similar_states(state)
            return self.similar_states_cache[state_key]
        else:
            # For rarely visited states, compute on demand (no cache)
            return self.similar_states(state)
    
    def generalized_qval(self, state, action, nsa):
        q = self.Q[state][action]
        if nsa >= 8:
            return q

        similar_states = self.get_cached_similar_states(state)
        # if no similar states, return from Q-table
        if not similar_states:
            return q

        # Compute the weighted average of Q-values from similar states
        weighted_sum = 0.0
        total_weight = 0.0

        weight = max(1.0, nsa)
        weighted_sum += q * weight
        total_weight += weight

        for sim_weight, sim_state, _ in similar_states:
            if sim_state in self.Q and self.Nsa[sim_state][action] > 0:
                sim_q = self.Q[sim_state][action]
                sim_nsa = self.Nsa[sim_state][action]
                weighted_sum += sim_q * sim_weight * sim_nsa * 0.25
                total_weight += sim_weight * sim_nsa * 0.25

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return q

    '''
    # Exploratory Q-Learning Agent, written utilizing pseudocode from AIMA textbook
    def q_learning_agent(self, percept):
        s_prime, r_prime = percept

        # if terminal state, then Q[s, None] <- r'
        if self.s is not None and self.is_terminal(self.s):
            self.Q[self.s][None] = r_prime
        
        # if s is not None increment Nsa[s,a]
        if self.s is not None:
            self.Nsa[self.s][self.a] += 1

            if not self.is_terminal(s_prime):
                # Compute max_a' Q[s',a'] with generalization
                max_q = float('-inf')
                for a in range(self.num_actions):
                    q_value = self.generalized_qval(s_prime, a, self.Nsa[s_prime][a])
                    if q_value > max_q:
                        max_q = q_value
            else:
                max_q = 0.0
            
            # Q[s,a] <- Q[s,a] + alpha(Nsa[s,a]) × (r + gamma × max_a' Q[s',a'] - Q[s,a])
            q_current = self.Q[self.s][self.a]
            alpha = self.get_alpha(self.Nsa[self.s][self.a])
            self.Q[self.s][self.a] = q_current + alpha * (self.r + self.gamma * max_q - q_current)
        
        if not self.is_terminal(s_prime):
            best_action = None
            best_value = float('-inf')

            # find argmax_a' f(Q[s',a'], Nsa[s',a'])
            for a in range(self.num_actions):
                q_value = self.generalized_qval(s_prime, a, self.Nsa[s_prime][a])
                exploration = self.exploration_function(q_value, self.Nsa[s_prime][a])

                if exploration > best_value:
                    best_value = exploration
                    best_action = a
            
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
            '''
    
    def q_learning_agent(self, percept):
        """Optimized Q-learning agent - compute generalized Q-values once per state"""
        s_prime, r_prime = percept
        self.step_counter += 1

        if self.s is not None and self.is_terminal(self.s):
            self.Q[self.s][None] = r_prime
        
        if self.s is not None:
            self.Nsa[self.s][self.a] += 1

            if not self.is_terminal(s_prime):
                # OPTIMIZATION: Compute all generalized Q-values once
                generalized_q_values = []
                for a in range(self.num_actions):
                    q_value = self.generalized_qval(s_prime, a, self.Nsa[s_prime][a])
                    generalized_q_values.append(q_value)
                
                max_q = max(generalized_q_values)
            else:
                max_q = 0.0
                generalized_q_values = None
            
            # Q-learning update (same as before)
            q_current = self.Q[self.s][self.a]
            self.Q[self.s][self.a] = q_current + self.alpha * (self.r + self.gamma * max_q - q_current)
        
        if not self.is_terminal(s_prime):
            best_action = None
            best_value = float('-inf')

            # OPTIMIZATION: Reuse computed Q-values if available
            if 'generalized_q_values' in locals() and generalized_q_values is not None:
                q_values = generalized_q_values
            else:
                # Compute once for action selection only
                q_values = []
                for a in range(self.num_actions):
                    q_value = self.generalized_qval(s_prime, a, self.Nsa[s_prime][a])
                    q_values.append(q_value)

            # Action selection using pre-computed values
            for a in range(self.num_actions):
                exploration = self.exploration_function(q_values[a], self.Nsa[s_prime][a])
                if exploration > best_value:
                    best_value = exploration
                    best_action = a
            
            self.prev_action = best_action  
            self.s = s_prime
            self.a = best_action
            self.r = r_prime
            self.prev_action = best_action  # Store for stuck detection
            return best_action
        else:
            self.prev_action = None
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

            while not self.is_terminal(current_state) and steps < 1000:
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
            if (episode + 1) % 10 == 0:
                self.save_qtable()

            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode > 0 else total_reward
                print(f"Episode {episode}: Reward = {total_reward}, Avg Last 10 = {avg_reward:.2f} for a3newpel.py")
                print(f"  Q-table size: {len(self.Q)}, Total state-action pairs: {sum(len(actions) for actions in self.Q.values())}")
        
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
    
    rewards = agent.run(num_episodes=10000)
    print("Episode rewards:", rewards)

if __name__ == "__main__":
    main()
