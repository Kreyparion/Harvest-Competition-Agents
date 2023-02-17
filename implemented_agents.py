from agents.random import RandomAgent
from agents.dqn import DQN_Agent
from agents.reinforce import Reinforce_Agent

agents_map = {
    "random": RandomAgent,
    "dqn": DQN_Agent,
    "reinforce": Reinforce_Agent,
}