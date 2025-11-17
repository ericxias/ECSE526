To run the agent, first install the ale-py environment with:
pip install ale-py
Then, to run the each file, run the following line:
python (agent_file).py
To see the Ms.Pacman display, change self.ale.setBool("display_screen", False) -> self.ale.setBool("display_screen", True)
When training the agent, set learning=True

To change the seed, uncomment #self.ale.setInt("random_seed", 123), and change the integer.
The agentperformance plots utilize random seeds.
To run agent performance, run the following line:
python agentperformance.py

To generate comparison graphs for generalization and exploration, run the following line:
python graphs.py

Optimistic Prior Exploration:
a3action.py -> Tabular Q-Learning Agent using Action-Approximation Generalization
a3distance.py -> Tabular Q-Learning Agent using Distance Metric Case Based Generalization
a3func.py -> Tabular Q-Learning Agent using Function Approximation Generalization
a3nongen.py -> Tabular Q-Learning Agent without Generalization

a3funcep.py -> Tabular Q-Learning Agent using Function Approximation Generalization with Epsilon Random Exploration
a3noexp.py -> Tabular Q-Learning Agent using Function Approximation Generalization without Exploration
