import matplotlib.pyplot as plt
import numpy as np
import time
import os
from a3newpel import MsPacmanAgent as DistanceMetricAgent

class MultiSeedAnalysis:
    def __init__(self, num_seeds=5, num_episodes=1000):
        self.num_seeds = num_seeds
        self.num_episodes = num_episodes
        self.seed_results = {}
        
    def run_agent_with_seed(self, agent_class, seed, agent_name):
        """Run an agent with a specific seed"""
        print(f"\n=== Running {agent_name} with Seed {seed} ===")
        
        # Create unique filename for each seed
        filename = f"seed_{seed}_{agent_name.lower().replace(' ', '_')}.pkl"
        agent = agent_class(file=filename)
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        episode_rewards = []
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            # Reset and run one episode
            agent.reset()
            reward = 0
            total_reward = 0
            current_state = agent.get_state()
            steps = 0
            previous_lives = agent.ale.lives()
            
            # Run episode
            while not agent.is_terminal(current_state) and steps < 1000:
                percept = (current_state, reward)
                a = agent.q_learning_agent(percept)
                if a is None:
                    break
                
                action = agent.legal_actions[a]
                raw_reward = agent.ale.act(action)
                
                # Add penalty for losing life
                penalty = 0
                current_lives = agent.ale.lives()
                if current_lives < previous_lives:
                    penalty = -10
                    previous_lives = current_lives
                
                reward = raw_reward + penalty
                current_state = agent.get_state()
                total_reward += raw_reward
                steps += 1
            
            # Final state update
            if agent.is_terminal(current_state):
                final_percept = (current_state, reward)
                agent.q_learning_agent(final_percept)
            
            episode_rewards.append(total_reward)

            # Print progress every 10 episodes
            if episode % 10 == 0:
                recent_avg = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
                elapsed_time = time.time() - start_time
                print(f"  Seed {seed}, Episode {episode}: Avg Reward = {recent_avg:.1f}, Time = {elapsed_time:.1f}s")
        
        total_time = time.time() - start_time
        final_performance = np.mean(episode_rewards[-100:])
        
        print(f"  Seed {seed} Completed: Final Performance = {final_performance:.1f}")
        
        return episode_rewards
    
    def run_multi_seed_analysis(self, agent_class, agent_name):
        """Run agent across multiple seeds"""
        print(f"\nRunning Multi-Seed Analysis for {agent_name}")
        print(f"Seeds: {self.num_seeds}, Episodes per seed: {self.num_episodes}")
        print("=" * 60)
        
        all_rewards = []
        
        for seed in range(self.num_seeds):
            seed_rewards = self.run_agent_with_seed(agent_class, seed, agent_name)
            all_rewards.append(seed_rewards)
            
            # Clean up to save memory
            import gc
            gc.collect()
        
        # Store results
        self.seed_results[agent_name] = {
            'all_rewards': all_rewards,
            'mean_rewards': np.mean(all_rewards, axis=0),
            'std_rewards': np.std(all_rewards, axis=0),
            'min_rewards': np.min(all_rewards, axis=0),
            'max_rewards': np.max(all_rewards, axis=0),
            'final_performances': [np.mean(rewards[-100:]) for rewards in all_rewards],
            'learning_curves': [self.calculate_learning_curve(rewards) for rewards in all_rewards]
        }
        
        return all_rewards
    
    def calculate_learning_curve(self, rewards, window=50):
        """Calculate smoothed learning curve"""
        learning_curve = []
        for i in range(len(rewards)):
            start_idx = max(0, i - window + 1)
            learning_curve.append(np.mean(rewards[start_idx:i+1]))
        return learning_curve
    
    def create_multi_seed_plots(self, agent_name):
        """Create comprehensive multi-seed analysis plots"""
        if agent_name not in self.seed_results:
            print(f"No data for {agent_name}")
            return
        
        data = self.seed_results[agent_name]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        episodes = np.arange(self.num_episodes)
        
        # Plot 1: Individual Seed Learning Curves
        ax1.set_title(f'{agent_name}: Learning Curves Across Seeds', fontsize=14)
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_seeds))
        
        for i, learning_curve in enumerate(data['learning_curves']):
            ax1.plot(episodes, learning_curve, 
                    label=f'Seed {i}', color=colors[i], alpha=0.7, linewidth=1)
        
        # Plot mean
        mean_learning_curve = np.mean(data['learning_curves'], axis=0)
        ax1.plot(episodes, mean_learning_curve, 
                label='Mean', color='black', linewidth=3)
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Average Reward (50-episode window)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean with Confidence Intervals
        ax2.set_title(f'{agent_name}: Mean Performance with Confidence Intervals', fontsize=14)
        
        mean_rewards = data['mean_rewards']
        std_rewards = data['std_rewards']
        
        # Calculate confidence intervals (mean ± 1.96*std/sqrt(n) for 95% CI)
        ci = 1.96 * std_rewards / np.sqrt(self.num_seeds)
        
        ax2.plot(episodes, mean_rewards, color='blue', linewidth=2, label='Mean')
        ax2.fill_between(episodes, mean_rewards - ci, mean_rewards + ci, 
                        alpha=0.3, color='blue', label='95% Confidence Interval')
        ax2.fill_between(episodes, data['min_rewards'], data['max_rewards'], 
                        alpha=0.1, color='gray', label='Min-Max Range')
        
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Game Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Final Performance Distribution
        ax3.set_title(f'{agent_name}: Final Performance Distribution', fontsize=14)
        final_performances = data['final_performances']
        
        ax3.hist(final_performances, bins=max(3, self.num_seeds//2), 
                alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(np.mean(final_performances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(final_performances):.1f}')
        ax3.axvline(np.median(final_performances), color='orange', linestyle='--', 
                   label=f'Median: {np.median(final_performances):.1f}')
        
        ax3.set_xlabel('Final Performance (Last 100 Episodes Average)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning Progress Statistics
        ax4.set_title(f'{agent_name}: Learning Progress Statistics', fontsize=14)
        
        # Calculate episodes to reach certain thresholds
        thresholds = [100, 200, 300, 400, 500]
        episodes_to_threshold = {thresh: [] for thresh in thresholds}
        
        for learning_curve in data['learning_curves']:
            for thresh in thresholds:
                # Find first episode where performance exceeds threshold
                exceeded = np.where(np.array(learning_curve) >= thresh)[0]
                if len(exceeded) > 0:
                    episodes_to_threshold[thresh].append(exceeded[0])
                else:
                    episodes_to_threshold[thresh].append(self.num_episodes)
        
        # Plot box plots for each threshold
        threshold_data = [episodes_to_threshold[thresh] for thresh in thresholds]
        box_plot = ax4.boxplot(threshold_data, labels=[f'{t}' for t in thresholds])
        
        ax4.set_xlabel('Performance Threshold')
        ax4.set_ylabel('Episodes to Reach Threshold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{agent_name.lower().replace(" ", "_")}_multi_seed_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_statistics(self, agent_name):
        """Create detailed summary statistics"""
        if agent_name not in self.seed_results:
            print(f"No data for {agent_name}")
            return
        
        data = self.seed_results[agent_name]
        
        print(f"\n{'='*80}")
        print(f"MULTI-SEED ANALYSIS SUMMARY: {agent_name}")
        print(f"{'='*80}")
        print(f"Number of seeds: {self.num_seeds}")
        print(f"Episodes per seed: {self.num_episodes}")
        
        # Final performance statistics
        final_performances = data['final_performances']
        print(f"\nFINAL PERFORMANCE STATISTICS (Last 100 Episodes):")
        print(f"  Mean: {np.mean(final_performances):.2f} ± {np.std(final_performances):.2f}")
        print(f"  Median: {np.median(final_performances):.2f}")
        print(f"  Min: {np.min(final_performances):.2f}")
        print(f"  Max: {np.max(final_performances):.2f}")
        print(f"  Coefficient of Variation: {np.std(final_performances)/np.mean(final_performances):.3f}")
        
        # Learning consistency
        print(f"\nLEARNING CONSISTENCY:")
        # Calculate how consistent learning is across seeds
        mean_curve = np.mean(data['learning_curves'], axis=0)
        std_curve = np.std(data['learning_curves'], axis=0)
        avg_cv = np.mean(std_curve / (mean_curve + 1e-8))  # Add small epsilon to avoid division by zero
        print(f"  Average CV across learning: {avg_cv:.3f}")
        
        # Episodes to reach milestones
        print(f"\nLEARNING MILESTONES (Mean ± Std):")
        thresholds = [200, 300, 400, 500]
        for thresh in thresholds:
            episodes_needed = []
            for learning_curve in data['learning_curves']:
                exceeded = np.where(np.array(learning_curve) >= thresh)[0]
                if len(exceeded) > 0:
                    episodes_needed.append(exceeded[0])
                else:
                    episodes_needed.append(self.num_episodes)
            
            if episodes_needed:
                mean_episodes = np.mean(episodes_needed)
                std_episodes = np.std(episodes_needed)
                success_rate = np.mean([e < self.num_episodes for e in episodes_needed])
                print(f"  Score {thresh}: {mean_episodes:.0f} ± {std_episodes:.0f} episodes (Success: {success_rate:.1%})")
        
        print(f"{'='*80}")

def main():
    # Configuration
    NUM_SEEDS = 5
    NUM_EPISODES = 1000
    
    print("MULTI-SEED GENERALIZATION ANALYSIS")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Seeds: {NUM_SEEDS}")
    print(f"  Episodes per seed: {NUM_EPISODES}")
    
    # Initialize analysis
    analysis = MultiSeedAnalysis(num_seeds=NUM_SEEDS, num_episodes=NUM_EPISODES)

    # Run analysis for Distance Metric Agent
    agent_name = "Distance Metric Agent"
    analysis.run_multi_seed_analysis(DistanceMetricAgent, agent_name)
    
    # Create plots and statistics
    analysis.create_multi_seed_plots(agent_name)
    analysis.create_summary_statistics(agent_name)
    
    print(f"\nAnalysis complete! Generated:")
    print(f"  - {agent_name.lower().replace(' ', '_')}_multi_seed_analysis.png")

if __name__ == "__main__":
    main()