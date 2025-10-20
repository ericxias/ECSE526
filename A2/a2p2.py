import numpy as np
from typing import List, Tuple, Dict

class KangarooWorld:
    def __init__(self):
        self.size = 9
        self.food = 5
        self.holes = {3, 7}
        self.gamma = 0.9
        self.valid_states = []
        for square in range(1, self.size + 1):
            for jump in range(-2, 3):
                if square in self.holes and jump != 0:
                    continue
                self.valid_states.append((square, jump))

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        square, jump = state
        if square in self.holes:
            return True
        if square == self.food and jump == 0:
            return True
        return False

    def get_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        if self.is_terminal(state):
            return []
        square, jump = state
        actions = []
        for action in range(-2, 3):
            if abs(action - jump) <= 1:
                actions.append(action)

        return actions
    
    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        square, _ = state
        next_square = square + action

        # Bouncing off the wall
        if next_square < 1:
            next_square = 1 - next_square
        elif next_square > self.size:
            next_square = self.size - (next_square - (self.size + 1))
        
        return (next_square, action)
        
    def get_reward(self, r: int, state: Tuple[int, int]) -> float:
        square, jump = state
        if square in self.holes:
            return -1.0
        
        if square == self.food and jump == 0:
            return 1.0
        
        return r
    
    def value_iteration():
        pass

    def policy_evaluation():
        pass

    def policy_iteration():
        pass

def main():
    pass



        