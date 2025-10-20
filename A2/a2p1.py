import numpy as np
from typing import Dict, Tuple, List, Set

class GridWorld:
    def __init__(self):
        self.size = 4
        self.gamma = 0.9
        self.epsilon = 0.0001
        
        # Terminal states and their rewards
        self.terminal_states = {(3, 4): -1, (4, 4): 1}
        
        # Blocked states
        self.blocked_states = {(3, 2), (1, 3)}
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = ['up', 'right', 'down', 'left']
        self.action_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Transition probabilities
        self.prob_forward = 0.7
        self.prob_left = 0.2
        self.prob_right = 0.1
        
        # Initialize utilities
        self.utilities = {}
        self.policy = {}
        
        # Initialize all valid states
        self.valid_states = set()
        for i in range(1, 5):
            for j in range(1, 5):
                if (i, j) not in self.blocked_states:
                    self.valid_states.add((i, j))
                    if (i, j) not in self.terminal_states:
                        self.utilities[(i, j)] = 0.0
                    else:
                        self.utilities[(i, j)] = self.terminal_states[(i, j)]
    
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if a state is valid (within bounds and not blocked)"""
        i, j = state
        return (1 <= i <= 4 and 1 <= j <= 4 and 
                state not in self.blocked_states)
    
    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next state given current state and action"""
        i, j = state
        di, dj = self.action_vectors[action]
        next_state = (i + di, j + dj)
        
        # If next state is invalid, stay in current state
        if not self.is_valid_state(next_state):
            return state
        return next_state
    
    def get_transition_probs(self, state: Tuple[int, int], action: int) -> Dict[Tuple[int, int], float]:
        """Get transition probabilities for all possible next states"""
        # terminal state -> stays in the same state
        if state in self.terminal_states:
            return {state: 1.0}
        
        transitions = {}
        
        # Forward movement, if next state is invalid, stays in the same state
        next_state = self.get_next_state(state, action)
        transitions[next_state] = transitions.get(next_state, 0) + self.prob_forward
        
        # Leftwards movement, if next state is invalid, stays in the same state
        left_action = (action - 1) % 4
        next_state = self.get_next_state(state, left_action)
        transitions[next_state] = transitions.get(next_state, 0) + self.prob_left

        # Rightwards movement, if next state is invalid, stays in the same state
        right_action = (action + 1) % 4
        next_state = self.get_next_state(state, right_action)
        transitions[next_state] = transitions.get(next_state, 0) + self.prob_right
        
        return transitions

    def get_reward(self, state: Tuple[int, int], r: float) -> float:
        """Get reward for being in a state"""
        if state in self.terminal_states:
            return self.terminal_states[state]
        return r
    
    def value_iteration(self, r: float) -> Dict:
        """Implement value iteration algorithm following Russell & Norvig pseudocode"""
        #print("Starting Value Iteration...")
        
        # Initialize U and U' vectors of utilities for states in S, initially zero
        U = {}
        U_prime = {}
        
        for state in self.valid_states:
            if state in self.terminal_states:
                U[state] = self.terminal_states[state]
                U_prime[state] = self.terminal_states[state]
            else:
                U[state] = 0.0
                U_prime[state] = 0.0
        
        iteration = 0
        epsilon_threshold = self.epsilon * (1 - self.gamma) / self.gamma
        
        while True:
            iteration += 1
            
            # U <- U'
            U = U_prime.copy()
            
            # δ, the maximum change in the utility of any state in an iteration
            delta = 0
            
            # for each state s in S do
            for state in self.valid_states:
                if state in self.terminal_states:
                    continue
                
                # Calculate max_a∈A(s) Σ P(s' | s, a)U(s')
                max_expected_utility = float('-inf')
                
                # Try all actions a in A(s)
                for action in range(4):
                    expected_utility = 0
                    transitions = self.get_transition_probs(state, action)
                    
                    # Σ P(s' | s, a)U(s')
                    for next_state, prob in transitions.items():
                        expected_utility += prob * U[next_state]
                    
                    max_expected_utility = max(max_expected_utility, expected_utility)
                
                # U'[s] <- R(s) + γ max_a∈A(s) Σ P(s' | s, a)U(s')
                U_prime[state] = self.get_reward(state, r) + self.gamma * max_expected_utility
                
                # if |U'[s] - U[s]| > δ then δ <- |U'[s] - U[s]|
                state_change = abs(U_prime[state] - U[state])
                if state_change > delta:
                    delta = state_change
            
            #print(f"Iteration {iteration}: δ = {delta:.6f}")
            
            # until δ < epsilon(1-γ)/γ
            if delta < epsilon_threshold:
                print(f"Converged after {iteration} iterations")
                break
        
        # return U
        return U_prime

    def policy_evaluation(self, r: float, policy: Dict, max_iterations: int = 100) -> Dict:
        """Evaluate a given policy"""
        utilities = {state: 0.0 for state in self.valid_states if state not in self.terminal_states}
        for state in self.terminal_states:
            utilities[state] = self.terminal_states[state]
        
        for _ in range(max_iterations):
            new_utilities = utilities.copy()
            
            for state in self.valid_states:
                if state in self.terminal_states:
                    continue
                
                action = policy[state]
                if action is None:
                    continue
                
                expected_utility = 0
                transitions = self.get_transition_probs(state, action)
                
                for next_state, prob in transitions.items():
                    expected_utility += prob * utilities[next_state]

                new_utilities[state] = self.get_reward(state, r) + self.gamma * expected_utility

            utilities = new_utilities
        
        return utilities
    
    def policy_iteration(self, r: float) -> Dict:
        """Implement policy iteration algorithm following Russell & Norvig pseudocode"""
        print("Starting Policy Iteration...")
        
        # U, a vector of utilities for states in S, initially zero
        U = {}
        for state in self.valid_states:
            if state in self.terminal_states:
                U[state] = self.terminal_states[state]
            else:
                U[state] = 0.0
        
        # pi, a policy vector indexed by state, initially random
        pi = {}
        for state in self.valid_states:
            if state in self.terminal_states:
                pi[state] = None
            else:
                pi[state] = 0  # Start with 'up' action
        
        iteration = 0
        
        while True:
            iteration += 1
            print(f"Policy Iteration {iteration}")
            
            # U <- POLICY-EVALUATION(pi, U, mdp)
            U = self.policy_evaluation(r, pi, max_iterations=100)
            
            # unchanged? <- true
            unchanged = True
            
            # for each state s in S do
            for state in self.valid_states:
                if state in self.terminal_states:
                    continue
                
                # Calculate max_a∈A(s) Σ P(s' | s, a)U[s']
                max_action_value = float('-inf')
                best_action = None
                
                for action in range(4):
                    action_value = 0
                    transitions = self.get_transition_probs(state, action)
                    
                    # Σ P(s' | s, a)U[s']
                    for next_state, prob in transitions.items():
                        action_value += prob * U[next_state]
                    
                    if action_value > max_action_value:
                        max_action_value = action_value
                        best_action = action
                
                # Calculate current policy value: Σ P(s'|s, pi[s]) U[s']
                current_policy_value = 0
                transitions = self.get_transition_probs(state, pi[state])
                for next_state, prob in transitions.items():
                    current_policy_value += prob * U[next_state]
                
                # if max_a∈A(s) Σ P(s' | s, a)U[s'] > Σ P(s'|s, pi[s]) U[s'] then do
                if max_action_value > current_policy_value:
                    # pi[s] <- argmax_a∈A(s) Σ P(s' | s, a)U[s']
                    pi[state] = best_action
                    # unchanged? <- false
                    unchanged = False
            
            # until unchanged?
            if unchanged:
                print(f"Policy converged after {iteration} iterations")
                break
        
        # return pi
        return pi

    def pi_policy(self, reward: float):
        # always go down
        states = [
            (4,1), (3,1), (2,1), (1,1), # u11, u12, u13, u14
            (4,2), (2,2), (1,2),        # u21, u23, u24
            (4,3), (3,3), (2,3),        # u31, u32, u33
            (4,4), (3,4), (2,4), (1,4)  # u41, u42, u43, u44
        ]

        n = len(states)
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        #print("Setting up linear equations for policy evaluation...")
    
        for index, state in enumerate(states):
            #print(f"Processing state {state} (index {index})")

            if state in self.terminal_states:
                A[index, index] = 1.0
                b[index] = self.terminal_states[state]
                #print(f"  Terminal state. Set U[{state}] = {b[index]}")
            else:
                #print(f"  Non-terminal state. Setting up equation for U[{state}]")
                # To make the linear system Au = b, rearrange the Bellman equation: U(s) - γ Σ P(s'|s,a) U(s') = R(s), where R(s) = reward
                b[index] = reward
                A[index, index] = 1.0
                # transition probabilities for down action
                transitions = self.get_transition_probs(state, 2)
                #print(f"  Transition probabilities for action 'down': {transitions}")
                for next_state, prob in transitions.items():
                    if next_state in states:
                        # check if next state is a terminal state
                        if next_state in self.terminal_states:
                            # the resulting probability γ P(s'|s,π) R(terminal) is added to b
                            b[index] += self.gamma * prob * self.terminal_states[next_state]
                            #print(f"    Next state {next_state} is terminal. Adding {self.gamma * prob * self.terminal_states[next_state]} to b[{index}]")
                        else:
                            next_index = states.index(next_state)
                            if next_state == state:
                                # staying in the same state, modify coefficient
                                A[index, index] -= self.gamma * prob
                                #print(f"    Next state {next_state} is the same as current state. Adjusting A[{index},{index}] to {A[index, index]}")
                            else:
                                A[index, next_index] -= self.gamma * prob
                                #print(f"    Next state {next_state} is different from current state. Adjusting A[{index},{next_index}] to {A[index, next_index]}")

                #print(f"  Final equation for state {state}: {A[index]} * U = {b[index]}")
        return A, b
    
    # Display logic generated by GitHub Copilot
    def display_linear_system_matrices(self):
        """Display the A matrix and b vector for both reward values"""
        
        print("\n" + "="*60)
        print("LINEAR SYSTEM ANALYSIS - 'Always Down' Policy")
        print("="*60)
        
        for r_val in [-0.2, -0.4]:
            print(f"\n{'='*20} r = {r_val} {'='*20}")
            
            A, b = self.pi_policy(r_val)

            # check if A is invertible
            det_A = np.linalg.det(A)
            if abs(det_A) < 1e-10:
                print("Matrix A is singular and cannot be inverted.")
            else:
                print(f"Matrix A is invertible (determinant = {det_A:.6f}).")

            print(f"\nMatrix A ({len(A)}x{len(A[0])}):")
            np.set_printoptions(precision=4, suppress=True, linewidth=200)
            print(A)
            
            print(f"\nVector b:")
            print(b)

    def calculate_cd(self):
        # We need two different r values to solve for c and d
        r1, r2 = -2.0, -1.0
        
        # Solve linear systems for both r values
        A1, b1 = self.pi_policy(r1)
        A2, b2 = self.pi_policy(r2)
        
        # Get utilities for both r values
        u1 = np.linalg.solve(A1, b1)  # utilities when r = -0.2
        u2 = np.linalg.solve(A2, b2)  # utilities when r = -0.4
        
        # Calculate d and c from the two solutions
        # From u = c + rd, we have two equations per state:
        # u1 = c + r1*d  and  u2 = c + r2*d
        d = (u2 - u1) / (r2 - r1)  # Slope of the line u vs r
        c = u1 - r1 * d            # y-intercept when r = 0
        
        print("\nLINEAR COEFFICIENTS for u = c + rd")
        
        print(f"State s32: c = {c[8]}, d = {d[8]}")
        print(f"State s21: c = {c[4]}, d = {d[4]}")
        print(f"State s44: c = {c[13]}, d = {d[13]}")
        
        return
    
    def value_iteration_graph(self):
        r_values = np.arange(-4.0, 0.1, 0.1)
        for r in r_values:
            utilities = self.value_iteration(r)
            u_32 = utilities[(3, 3)]
            u_33 = utilities[(2, 3)]
            u_23 = utilities[(2, 2)]
            print(f"r: {r: .1f}, u_32: {u_32}, u_33: {u_33}, u_23: {u_23}")
        
def main():
    """Main function to run both algorithms"""
    world = GridWorld()
    
    # Run Value Iteration - get utilities
    value_iteration = world.value_iteration(-0.04)
    
    # Run Policy Iteration - get optimal policy
    policy_iteration = world.policy_iteration(-0.04)
    
    # Result display logic generated by Github Copilot
    print("\nVALUE ITERATION Results:")
    print("=" * 50)
    print("\nState Utilities:")
    print("Row\\Col    1        2        3        4")
    for i in range(1, 5):
        row_str = f"  {i}   "
        for j in range(1, 5):
            if (i, j) in world.blocked_states:
                row_str += "  BLOCK  "
            else:
                row_str += f"{value_iteration[(i, j)]:8.4f} "
        print(row_str)
    
    print("\nPOLICY ITERATION Results:")
    print("=" * 50)
    print("\nOptimal Policy:")
    print("Row\\Col  1   2   3   4")
    action_symbols = ['↑', '→', '↓', '←']
    for i in range(1, 5):
        row_str = f"  {i}   "
        for j in range(1, 5):
            if (i, j) in world.blocked_states:
                row_str += " ■  "
            elif (i, j) in world.terminal_states:
                row_str += " T  "
            else:
                action = policy_iteration[(i, j)]
                if action is not None:
                    row_str += f" {action_symbols[action]}  "
                else:
                    row_str += " -  "
        print(row_str)

    world.display_linear_system_matrices()
    world.calculate_cd()
    #world.value_iteration_graph()


if __name__ == "__main__":
    main()