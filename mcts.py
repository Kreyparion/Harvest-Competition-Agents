from env.utils import Action, State
from env.environnement import Env
from copy import deepcopy
import numpy as np
import time



class MCTS:
    def __init__(self,env:Env,parent=None,parent_action=None):
        self.env = env
        self.depth = self.env.state.step_num
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = []
        self._untried_actions = self.env.getPossibleActionsAsInt()
        return

    def q(self):
        return sum(self._results)
    
    def n(self):
        return self._number_of_visits
    
    def expand(self):
        action = self._untried_actions.pop()
        children_env = deepcopy(self.env)
        _ = children_env.step(action)
        child_node = MCTS(children_env,parent=self)
        #if self.children != []:
        #    print(self.depth,self.children[0].depth,child_node.depth)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        if self.depth >= self.env.max_step - 1:
            return True
        return False
    
    def rollout(self,agent_for_rollout=None):
        current_rollout_state = deepcopy(self.env)
        while current_rollout_state.state.step_num != self.env.max_step:
            possible_moves = current_rollout_state.getPossibleActionsAsInt()
            action = self.rollout_policy(possible_moves,current_rollout_state,agent_for_rollout)
            current_rollout_state.step(action)
        return current_rollout_state.score,current_rollout_state
    
    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results.append(result)
        if self.parent:
            self.parent.backpropagate(result)
    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    def best_child(self, c_param=100):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def rollout_policy(self, possible_moves,env,agent_for_rollout=None):
        if agent_for_rollout == None:
            return possible_moves[np.random.randint(len(possible_moves))]
        else:
            return agent_for_rollout.act(env.state,training=False)
    
    

class MCTSearch:
    def __init__(self,node,agent_for_rollout=None) -> None:
        self.root = node
        self.agent_for_rollout = agent_for_rollout

    def _tree_policy(self):
        current_node = self.root
        #print(current_node.is_terminal_node(),current_node.q(),current_node._untried_actions)
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
            #print(current_node._untried_actions,current_node.depth)
        return current_node
    
    def best_action(self):
        simulation_no = 15000
        best_reward = 0
        max_depth = 0
        best_result = None
        for i in range(simulation_no):
            
            v = self._tree_policy()
            reward,result = v.rollout(self.agent_for_rollout)
            v.backpropagate(reward)
            if reward > best_reward:
                best_reward = reward
                best_result = result
                print(best_reward)
            if v.depth > max_depth:
                max_depth = v.depth
                print(max_depth)
            if i % 1000 == 0:
                print(str(int(i*10000/simulation_no)/100)+"%")
        return best_result

def run_mcts(agent_for_rollout=None):
    env = Env(42)
    env.reset()
    root = MCTS(env)
    search = MCTSearch(root,agent_for_rollout=agent_for_rollout)
    best_result = search.best_action()
    best_result.save_solution()
    print(best_result.score)
    return

