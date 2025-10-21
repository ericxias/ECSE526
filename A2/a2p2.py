import numpy as np
from typing import List, Tuple, Dict

class Kangaroo:
    def __init__(self):
        self.size = 9
        self.food = 5
        self.holes = {3, 7}
        self.gamma = 0.9
        self.valid_states = []
        self.epsilon = 0.0001
        # Generate all valid states (square, j(t-1))
        for square in range(1, self.size + 1):
            for jump in range(-2, 3):
                self.valid_states.append((square, jump))

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        square, jump = state
        if square in self.holes:
            return True
        if square == self.food and jump == 0:
            return True
        return False

    def get_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        # Return empty list for terminal states
        if self.is_terminal(state):
            return []
        square, jump = state
        actions = []
        # Check |j(t) - j(t-1)| <= 1
        for action in range(-2, 3):
            if abs(action - jump) <= 1:
                actions.append(action)

        return actions
    
    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        square, _ = state
        next_square = square + action

        # Bouncing off the wall
        # Left wall
        if next_square < 1:
            next_square = 1 - next_square
        # Right wall
        elif next_square > self.size:
            next_square = 2* self.size - next_square + 1
        
        return (next_square, action)
        
    def get_reward(self, r: int, state: Tuple[int, int]) -> float:
        square, jump = state
        if square in self.holes:
            return -1.0
        
        if square == self.food and jump == 0:
            return 1.0
        
        return r
    
    # Value Iteration logic written utilizing pseudocode from AIMA textbook
    def value_iteration(self, r: float) -> Dict:
        # Initialize U and U', terminal states to their rewards
        U = {}
        U_prime = {}
        
        for state in self.valid_states:
            if self.is_terminal(state):
                U[state] = self.get_reward(r, state)
                U_prime[state] = self.get_reward(r, state)
            else:
                U[state] = 0.0
                U_prime[state] = 0.0
        
        while True:
            # U ← U', δ ← 0
            U = U_prime.copy()
            delta = 0
            # for each state s in S do
            for s in self.valid_states:
                if self.is_terminal(s):
                    continue

                actions = self.get_valid_actions(s)
                # if no valid actions, skip
                if not actions:
                    continue

                max_utility = float('-inf')
                # U'[s] ← R(s) + γ max_a Σ P(s'|s,a)U[s']
                # Deterministic transition, try each action j(t) and select max utility
                for a in actions:
                    s_prime = self.get_next_state(s, a)
                    utility = self.get_reward(r, s) + self.gamma*U[s_prime]
                    max_utility = max(max_utility, utility)
                U_prime[s] = max_utility
                
                # if |U'[s] - U[s]| > δ then δ ← |U'[s] - U[s]|
                if abs(U_prime[s] - U[s]) > delta:
                    delta = abs(U_prime[s] - U[s])
                
            # until δ < ε(1 - γ)/γ
            if delta < self.epsilon * (1 - self.gamma) / self.gamma:
                break
        
        return U_prime


    def policy_evaluation(self, policy: Dict, U: Dict, r: float) -> Dict:
        # Simplified Bellman equation for policy evaluation Ui(s) = R(s) + γ s'Σ P(s'| s, pi i(s))Ui(s') = R(s) + γ Ui(s')
        while True:
            U_prime = {}
            delta = 0.0
            for s in self.valid_states:
                if self.is_terminal(s):
                    U_prime[s] = U[s]
                    continue

                a = policy.get(s)
                s_prime = self.get_next_state(s, a)
                U_prime[s] = self.get_reward(r, s) + self.gamma * U[s_prime]
                if abs(U_prime[s] - U[s]) > delta:
                    delta = abs(U_prime[s] - U[s])
                    
            U = U_prime
            if delta < self.epsilon * (1 - self.gamma) / self.gamma:
                break
        return U
        
    # Policy Iteration logic written utilizing pseudocode from AIMA textbook
    def policy_iteration(self, r: float) -> Dict:
        # Initialize local variables pi and U
        U = {}
        pi = {}

        for state in self.valid_states:
            if self.is_terminal(state):
                U[state] = self.get_reward(r, state)
            else:
                U[state] = 0.0
                
        for state in self.valid_states:
            if self.is_terminal(state):
                pi[state] = None
            else:
                actions = self.get_valid_actions(state)
                pi[state] = actions[0]  
        
        while True:
            # U <- Policy-Evaluation, unchanged <- true
            U = self.policy_evaluation(pi, U, r)
            unchanged = True
            # for each state s in S do
            for s in self.valid_states:
                if self.is_terminal(s):
                    continue

                actions = self.get_valid_actions(s)
                max_policy = float('-inf')
                best_action = None
                # Deterministic transition, try each action and select max policy
                # Calculate max_a Σ P(s'|s,a)U[s'] -> max U[s'] available
                for a in actions:
                    s_prime = self.get_next_state(s, a)
                    policy = self.get_reward(r, s) + self.gamma * U[s_prime]

                    if policy > max_policy:
                        max_policy = policy
                        best_action = a


                # Calculate current policy value: Σ P(s'|s, pi[s]) U[s'] -> Utility of current policy action
                current_action = pi[s]
                if current_action is not None:
                    next_state = self.get_next_state(s, current_action)
                    current_policy = self.get_reward(r, s) + self.gamma * U[next_state]
                else:
                    current_policy = float('-inf')

                # if max_a Σ P(s'|s,a)U[s'] > Σ P(s'|s, pi[s]) U[s'] then
                if max_policy > current_policy:
                    # pi[s] ← argmax_a Σ P(s'|s,a)U[s']
                    pi[s] = best_action
                    # unchanged ← false
                    unchanged = False
                    
            # until unchanged
            if unchanged:
                break

        return pi
    
    # Print logic generated by Github Copilot
    def print_available_actions(self):
        """Print available actions for each state"""
        print("Available Actions for Each State:")
        print("Position | Prev Jump | Available Actions")
        print("-" * 50)
        
        for position in range(1, self.size + 1):
            for prev_jump in range(-2, 3):
                state = (position, prev_jump)
                if state not in self.valid_states:
                    continue
                    
                actions = self.get_valid_actions(state)
                print(f"    {position}    |     {prev_jump:2d}    | {actions}")
    
    def print_results(self, utilities: Dict, policy: Dict, r: float):
        """Print utilities and policy in a readable format"""
        print(f"\nResults (r = {r}):")
        print("=" * 80)
        
        print("\nState Utilities and Optimal Policy:")
        print("Position | Prev Jump | Utility   | Optimal Action | End Position | State Type")
        print("-" * 85)
        
        for position in range(1, self.size + 1):
            for prev_jump in range(-2, 3):
                state = (position, prev_jump)
                if state not in self.valid_states:
                    continue
                    
                utility = utilities.get(state, 0.0)
                action = policy.get(state, "None") if policy else "N/A"
                
                # Calculate end position after taking the optimal action j(t)
                end_position = "N/A"
                if action is not None and action != "None" and not self.is_terminal(state):
                    try:
                        next_state = self.get_next_state(state, action)
                        end_position = str(next_state[0])
                    except:
                        end_position = "Error"
                
                # Determine state type
                if position in self.holes:
                    state_type = "HOLE"
                    if policy:
                        action = "Terminal"
                    end_position = "N/A"
                elif position == self.food and prev_jump == 0:
                    state_type = "FOOD"
                    if policy:
                        action = "Terminal"
                    end_position = "N/A"
                elif position == self.food:
                    state_type = "Food (moving)"
                else:
                    state_type = "Normal"
                
                print(f"    {position}    |     {prev_jump:2d}    | {utility:8.4f}  | {str(action):>6}        | {str(end_position):>6}      | {state_type}")
def main():
    r_values = [-0.04, -0.5, -1.0]

    kangaroo = Kangaroo()
    # Actions for question 3
    kangaroo.print_available_actions()

    for r in r_values:
        print(f"\n{'='*70}")
        print(f"SOLVING KANGAROO WORLD FOR r = {r}")
        print(f"{'='*70}")
        
        utilities = kangaroo.value_iteration(r)
        policies = kangaroo.policy_iteration(r)

        
        kangaroo.print_results(utilities, policies, r)
        
if __name__ == "__main__":
    main()



        