# Ms Pacman Agent
from ale_py import ALEInterface, roms
import random
from collections import defaultdict
import numpy as np

class MsPacmanAgent:
    def __init__(self):
        self.ale = ALEInterface()
        self.ale.loadROM(roms.get_rom_path("MSPACMAN.bin"))
        self.ale.setInt("random_seed", 123)
        self.ale.setBool("display_screen", True)
        
        self.legal_actions = self.ale.getLegalActionSet()
        self.Q = defaultdict(lambda: defaultdict(float))
        self.Nsa = defaultdict(lambda: defaultdict(int))
        self.s = None
        self.a = None
        self.r = None
        self.gamma = 0.99  
        self.exploration_factor = 2.0  

    def get_state(self):
        ram = self.ale.getRAM()
        important_ram = ram[10:20]
        return tuple(important_ram)
    
    def is_terminal(self):
        return self.ale.game_over()
    
    def get_alpha(self, nsa):
        if nsa == 0:
            return 1.0
        else:
            return 1.0 / np.sqrt(nsa)

    def exploration_function(self, q_value, nsa):
        pass

    def q_learning_agent(self, percept):
        pass

    def run(self, num_episodes=100):
        pass

    def reset(self):
        self.ale.reset_game()
        self.s = None
        self.a = None
        self.r = None
        return
