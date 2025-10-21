import numpy as np
from typing import Dict, Tuple, List, Set

class Grid:
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
        
        # Initialize all valid states
        self.utilities = {}
        self.valid_states = set()
        for i in range(1, 5):
            for j in range(1, 5):
                if (i, j) not in self.blocked_states:
                    self.valid_states.add((i, j))
                    if (i, j) not in self.terminal_states:
                        self.utilities[(i, j)] = 0.0
                    else:
                        self.utilities[(i, j)] = self.terminal_states[(i, j)]
    
    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        i, j = state
        di, dj = self.action_vectors[action]
        next_state = (i + di, j + dj)
        
        # If next state is invalid, stay in current state
        if 1 <= i + di <= 4 and 1 <= j + dj <= 4 and next_state not in self.blocked_states:
            return next_state
        return state
    
    def get_transition_probs(self, state: Tuple[int, int], move: int) -> Dict[Tuple[int, int], float]:
        # returns state transition probabilities P(s' | s, a)
        # terminal state -> stays in the same state
        if state in self.terminal_states:
            return {state: 1.0}
        
        transitions = {}
        
        # Forward movement, if next state is invalid, stays in the same state
        next_state = self.get_next_state(state, move)
        transitions[next_state] = transitions.get(next_state, 0) + self.prob_forward
        
        # Leftwards movement, if next state is invalid, stays in the same state
        left_move = (move - 1) % 4
        next_state = self.get_next_state(state, left_move)
        transitions[next_state] = transitions.get(next_state, 0) + self.prob_left

        # Rightwards movement, if next state is invalid, stays in the same state
        right_move = (move + 1) % 4
        next_state = self.get_next_state(state, right_move)
        transitions[next_state] = transitions.get(next_state, 0) + self.prob_right
        
        return transitions

    def get_reward(self, state: Tuple[int, int], r: float) -> float:
        if state in self.terminal_states:
            return self.terminal_states[state]
        return r
    
    def value_iteration(self, r: float) -> Dict:        
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

        while True:
            # U <- U', δ <- 0
            U = U_prime.copy()
            delta = 0   
            # for each state s in S do
            for s in self.valid_states:
                if s in self.terminal_states:
                    continue
                
                max_utility = float('-inf')
                # max_a Σ P(s' | s, a)U(s')
                # Try all actions a in A(s)
                for a in range(4):
                    utility = 0
                    transitions = self.get_transition_probs(s, a)

                    # Σ P(s' | s, a)U(s')
                    for s_prime, prob in transitions.items():
                        utility += prob * U[s_prime]
                    max_utility = max(max_utility, utility)
                # U'[s] <- R(s) + γ max_a Σ P(s' | s, a)U(s')
                U_prime[s] = self.get_reward(s, r) + self.gamma * max_utility
                
                # if |U'[s] - U[s]| > δ then δ ← |U'[s] - U[s]|
                if abs(U_prime[s] - U[s]) > delta:
                    delta = abs(U_prime[s] - U[s])
                        
            # until δ < epsilon(1-γ)/γ
            if delta < self.epsilon * (1 - self.gamma) / self.gamma:
                break
        
        return U_prime

    def policy_evaluation(self, r: float, policy: Dict, U: Dict) -> Dict:
        # Simplified Bellman equation for policy evaluation Ui(s) = R(s) + γ s'Σ P(s'| s, pi i(s))Ui(s') = R(s) + γ Ui(s')
        while True:
            U_prime = {}
            delta = 0.0

            for s in self.valid_states:
                if s in self.terminal_states:
                    U_prime[s] = self.terminal_states[s]
                    continue

                a = policy.get(s)
                utility = 0
                transitions = self.get_transition_probs(s, a)

                # U(s) = R(s) + γ Σ P(s'|s,π(s)) U(s')
                for s_prime, prob in transitions.items():
                    utility += prob * U[s_prime]

                U_prime[s] = self.get_reward(s, r) + self.gamma * utility

                # if |U'[s] - U[s]| > δ then δ ← |U'[s] - U[s]|
                if abs(U_prime[s] - U[s]) > delta:
                    delta = abs(U_prime[s] - U[s])

            U = U_prime

            # until δ < epsilon(1-γ)/γ
            if delta < self.epsilon * (1 - self.gamma) / self.gamma:
                break
        return U
                
                
    
    def policy_iteration(self, r: float) -> Dict:
        # Initialize local variables pi and U
        U = {}
        pi = {}

        for state in self.valid_states:
            if state in self.terminal_states:
                U[state] = self.terminal_states[state]
            else:
                U[state] = 0.0

        for state in self.valid_states:
            if state in self.terminal_states:
                pi[state] = None
            else:
                pi[state] = 0  # Start with 'up' action
                
        while True:
            # U <- Policy-Evaluation, unchanged <- true
            U = self.policy_evaluation(r, pi, U)
            unchanged = True
            # for each state s in S do
            for s in self.valid_states:
                if s in self.terminal_states:
                    continue
                            
                max_policy = float('-inf')
                best_action = None
                # Calculate max_a Σ P(s' | s, a)U[s']
                for action in range(4):
                    action_value = 0
                    transitions = self.get_transition_probs(s, action)
                    
                    # max_a Σ P(s' | s, a)U[s']
                    for s_prime, prob in transitions.items():
                        action_value += prob * U[s_prime]

                    if action_value > max_policy:
                        max_policy = action_value
                        best_action = action
                
                # Calculate current policy value: Σ P(s'|s, pi[s]) U[s']
                current_policy = 0
                transitions = self.get_transition_probs(s, pi[s])
                for s_prime, prob in transitions.items():
                    current_policy += prob * U[s_prime]

                # if max_a∈A(s) Σ P(s' | s, a)U[s'] > Σ P(s'|s, pi[s]) U[s'] then do
                if max_policy > current_policy:
                    # pi[s] <- max_a Σ P(s' | s, a)U[s']
                    pi[s] = best_action
                    # unchanged? <- false
                    unchanged = False
            
            # until unchanged?
            if unchanged:
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
        # Solve for c and d with two different reward values
        r1, r2 = -2.0, -1.0
        
        # Solve linear systems for both r values
        A1, b1 = self.pi_policy(r1)
        A2, b2 = self.pi_policy(r2)
        
        # Get utilities for both r values
        u1 = np.linalg.solve(A1, b1)  
        u2 = np.linalg.solve(A2, b2)  
        
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
    
    def value_iteration_values(self):
        r_values = np.arange(-4.0, 0.1, 0.1)
        for r in r_values:
            utilities = self.value_iteration(r)
            u_32 = utilities[(3, 3)]
            u_33 = utilities[(2, 3)]
            u_23 = utilities[(2, 2)]
            print(f"r: {r: .1f}, u_32: {u_32}, u_33: {u_33}, u_23: {u_23}")
        
def main():
    """Main function to run both algorithms"""
    grid = Grid()
    
    # Run Value Iteration - get utilities
    value_iteration = grid.value_iteration(-0.04)
    
    # Run Policy Iteration - get optimal policy
    policy_iteration = grid.policy_iteration(-0.04)

    # Result display logic generated by Github Copilot
    print("\nVALUE ITERATION Results:")
    print("=" * 50)
    print("\nState Utilities:")
    print("Row\\Col    1        2        3        4")
    for i in range(1, 5):
        row_str = f"  {i}   "
        for j in range(1, 5):
            if (i, j) in grid.blocked_states:
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
            if (i, j) in grid.blocked_states:
                row_str += " ■  "
            elif (i, j) in grid.terminal_states:
                row_str += " T  "
            else:
                action = policy_iteration[(i, j)]
                if action is not None:
                    row_str += f" {action_symbols[action]}  "
                else:
                    row_str += " -  "
        print(row_str)

    grid.display_linear_system_matrices()
    grid.calculate_cd()
    #grid.value_iteration_values()


if __name__ == "__main__":
    main()