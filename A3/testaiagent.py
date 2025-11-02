# Ms Pacman Agent
from ale_py import ALEInterface, roms
import random
import numpy as np
from collections import defaultdict

class MsPacmanAgent:
    def __init__(self):
        self.ale = ALEInterface()
        self.ale.loadROM(roms.get_rom_path("MSPACMAN.bin"))
        self.ale.setInt("random_seed", 123)
        self.ale.setBool("display_screen", True)
        
        # Get legal actions
        self.legal_actions = self.ale.getLegalActionSet()
        self.num_actions = len(self.legal_actions)
        
        # Q-Learning Agent persistent variables
        self.Q = defaultdict(lambda: defaultdict(float))  # Q[s][a] = value
        self.Nsa = defaultdict(lambda: defaultdict(int))  # Nsa[s][a] = frequency
        self.s = None  # previous state
        self.a = None  # previous action
        self.r = None  # previous reward
        
        # Q-Learning parameters
        self.gamma = 0.99  # discount factor
        self.exploration_factor = 2.0  # for exploration function
        
    def get_state(self):
        """Simple state representation"""
        # Get RAM state (simpler than screen processing)
        ram = self.ale.getRAM()
        # Use a subset of RAM values that are likely important for Ms. Pacman
        # This is a simplified state representation
        important_ram = ram[10:20]  # arbitrary subset for demo
        return tuple(important_ram)
    
    def is_terminal(self, state):
        """Check if state is terminal"""
        return self.ale.game_over()
    
    def alpha(self, nsa):
        """Learning rate function - decreases with frequency"""
        return 1.0 / (1.0 + nsa)
    
    def exploration_function(self, q_value, nsa):
        """Exploration function f(Q[s',a'], Nsa[s',a'])"""
        if nsa < self.exploration_factor:
            return float('inf')  # Encourage exploration of unvisited state-action pairs
        return q_value
    
    def q_learning_agent(self, percept):
        """
        Q-Learning Agent following the given pseudocode
        
        Args:
            percept: tuple of (current_state, reward)
        
        Returns:
            action: the action to take
        """
        s_prime, r_prime = percept
        
        # if Terminal(s) then Q[s, None] <- r'
        if self.s is not None and self.is_terminal(self.s):
            self.Q[self.s][None] = r_prime
        
        # if s is not null then
        if self.s is not None:
            # increment Nsa[s,a]
            self.Nsa[self.s][self.a] += 1
            
            # Q[s,a] <- Q[s,a] + alpha(Nsa[s,a])(r + gamma*max_a' Q[s', a'] - Q[s,a])
            if not self.is_terminal(s_prime):
                # Find max_a' Q[s', a']
                max_q_next = max([self.Q[s_prime][action_idx] 
                                for action_idx in range(self.num_actions)], 
                               default=0.0)
            else:
                max_q_next = 0.0
            
            # Q-learning update
            current_q = self.Q[self.s][self.a]
            alpha_val = self.alpha(self.Nsa[self.s][self.a])
            target = self.r + self.gamma * max_q_next
            self.Q[self.s][self.a] = current_q + alpha_val * (target - current_q)
        
        # s, a, r <- s', argmax_a' f(Q[s', a'], Nsa[s', a']), r'
        if not self.is_terminal(s_prime):
            # argmax_a' f(Q[s', a'], Nsa[s', a'])
            best_action = None
            best_value = float('-inf')
            
            for action_idx in range(self.num_actions):
                q_val = self.Q[s_prime][action_idx]
                nsa_val = self.Nsa[s_prime][action_idx]
                exploration_val = self.exploration_function(q_val, nsa_val)
                
                if exploration_val > best_value:
                    best_value = exploration_val
                    best_action = action_idx
            
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
    
    def reset_episode(self):
        """Reset for new episode"""
        self.ale.reset_game()
        self.s = None
        self.a = None
        self.r = None
        return self.get_state()
    
    def run_q_learning(self, num_episodes=100):
        """Run Q-learning for specified number of episodes"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Reset game and agent state
            initial_state = self.reset_episode()
            total_reward = 0
            step_count = 0
            
            # Initial percept (state, reward=0)
            current_state = initial_state
            current_reward = 0
            
            while not self.ale.game_over() and step_count < 10000:  # Max steps per episode
                # Get action from Q-learning agent
                percept = (current_state, current_reward)
                action_idx = self.q_learning_agent(percept)
                
                if action_idx is None:  # Terminal state
                    break
                
                # Execute action in environment
                action = self.legal_actions[action_idx]
                reward = self.ale.act(action)
                
                # Get next state
                next_state = self.get_state()
                
                # Update for next iteration
                current_state = next_state
                current_reward = reward
                total_reward += reward
                step_count += 1
            
            # Handle final state
            if self.ale.game_over():
                final_percept = (current_state, current_reward)
                self.q_learning_agent(final_percept)
            
            episode_rewards.append(total_reward)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode > 0 else total_reward
                print(f"Episode {episode}: Reward = {total_reward}, Avg Last 10 = {avg_reward:.2f}")
                print(f"  Q-table size: {len(self.Q)}, Total state-action pairs: {sum(len(actions) for actions in self.Q.values())}")
        
        return episode_rewards
    
    def print_q_statistics(self):
        """Print some statistics about learned Q-values"""
        total_states = len(self.Q)
        total_sa_pairs = sum(len(actions) for actions in self.Q.values())
        
        print(f"\n=== Q-Learning Statistics ===")
        print(f"Total states visited: {total_states}")
        print(f"Total state-action pairs: {total_sa_pairs}")
        
        if total_states > 0:
            # Show some example Q-values
            print("\nSample Q-values:")
            for i, (state, actions) in enumerate(list(self.Q.items())[:5]):
                print(f"State {i}: {dict(actions)}")
        
        # Show most visited state-action pairs
        all_visits = []
        for state, actions in self.Nsa.items():
            for action, count in actions.items():
                all_visits.append((count, state, action))
        
        all_visits.sort(reverse=True)
        print(f"\nMost visited state-action pairs:")
        for count, state, action in all_visits[:5]:
            q_val = self.Q[state][action]
            print(f"  State: {str(state)[:50]}..., Action: {action}, Visits: {count}, Q-value: {q_val:.3f}")

# Example usage
if __name__ == "__main__":
    agent = MsPacmanAgent()
    
    print("Starting Q-Learning Agent for Ms. Pacman...")
    print(f"Legal actions: {agent.legal_actions}")
    print(f"Number of actions: {agent.num_actions}")
    
    # Run Q-learning
    rewards = agent.run_q_learning(num_episodes=50)
    
    # Print final statistics
    agent.print_q_statistics()
    
    print(f"\nFinal average reward over last 10 episodes: {np.mean(rewards[-10:]):.2f}")