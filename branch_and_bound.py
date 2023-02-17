from env.utils import Action, State
from env.environnement import Env
import sys
import numpy as np
import time
import copy
import random
import json

class BranchAndBound:
    #Add penalty to invalid solutions
    def __init__(self, solution:Env,use_json=False):
        self.solution = solution
        self.max_depth = solution.max_step
        self.best_cost = 0
        self.best_sol = None
        self.possible_action = self.solution.getPossibleActionsAsInt()
        self.n_children = len(self.possible_action)
        self.cost_at_depth = dict()
        self.time_start = time.time()
        self.time_in = 10000
        if not use_json:
            for i in range(self.max_depth):
                self.cost_at_depth[i] = 0
        else:
            f = open('best_at_depth.json')
            data = json.load(f)
            for k,v in data.items():
                self.cost_at_depth[int(k)] = v
                


    def run(self):
        self.__branch_and_bound__(self.solution)
        return self.best_sol
        
    def __branch_and_bound__(self, solution:Env):
        if time.time()-self.time_start > self.time_in:
            return
        if solution.step_num == self.max_depth:
            return
        #Compute the cost of the solution
        cost = solution.score
        #If the solution is valid, update the best solution
        depth = solution.step_num

        if cost > self.best_cost:
            self.best_cost = cost
            self.best_sol = solution

        #If the cost is lower than the best solution, stop the exploration
        if (depth == 0 or cost >= self.cost_at_depth[depth-1]) and (self.cost_at_depth[depth] == 0 or cost > self.cost_at_depth[depth]):
            self.cost_at_depth[depth] = cost
            with open('best_at_depth.json','w+') as fp:
                json.dump(self.cost_at_depth,fp)
            print("Depth: ", depth, " - Cost: ", cost, " - Best cost: ", self.best_cost)
        
        
        #elif depth <= 15:
        #    pass
        #elif depth <= 50 and (cost >= self.cost_at_depth[depth]*0.8 and cost >= self.cost_at_depth[depth-1]*0.84 and cost >= self.cost_at_depth[depth-2]*0.86 and cost >= self.cost_at_depth[depth-3]*0.9 and cost >= self.cost_at_depth[depth-4]*0.92 and cost >= self.cost_at_depth[depth-5]*0.94 and cost >= self.cost_at_depth[depth-6]*0.952 and cost >= self.cost_at_depth[depth-7]*0.96 and cost >= self.cost_at_depth[depth-8]*0.97):
        #    pass
        #elif depth <= 80 and (cost >= self.cost_at_depth[depth]*0.86 and cost >= self.cost_at_depth[depth-1]*0.9 and cost >= self.cost_at_depth[depth-2]*0.92 and cost >= self.cost_at_depth[depth-3]*0.94 and cost >= self.cost_at_depth[depth-4]*0.952 and cost >= self.cost_at_depth[depth-5]*0.96 and cost >= self.cost_at_depth[depth-6]*0.97 and cost >= self.cost_at_depth[depth-7]*0.98 and cost >= self.cost_at_depth[depth-8]*0.99):
        #    pass
        #elif depth <= 100 and (cost >= self.cost_at_depth[depth]*0.92 and cost >= self.cost_at_depth[depth-1]*0.94 and cost >= self.cost_at_depth[depth-2]*0.952 and cost >= self.cost_at_depth[depth-3]*0.96 and cost >= self.cost_at_depth[depth-4]*0.97 and cost >= self.cost_at_depth[depth-5]*0.98 and cost >= self.cost_at_depth[depth-6]*0.99 and cost >= self.cost_at_depth[depth-7]*0.993 and cost >= self.cost_at_depth[depth-8]*0.995 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 200 and (cost >= self.cost_at_depth[depth]*0.94 and cost >= self.cost_at_depth[depth-1]*0.952 and cost >= self.cost_at_depth[depth-2]*0.96 and cost >= self.cost_at_depth[depth-3]*0.97 and cost >= self.cost_at_depth[depth-4]*0.98 and cost >= self.cost_at_depth[depth-5]*0.99 and cost >= self.cost_at_depth[depth-6]*0.993 and cost >= self.cost_at_depth[depth-7]*0.995 and cost >= self.cost_at_depth[depth-8]*0.998 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 300 and (cost >= self.cost_at_depth[depth]*0.95 and cost >= self.cost_at_depth[depth-1]*0.96 and cost >= self.cost_at_depth[depth-2]*0.97 and cost >= self.cost_at_depth[depth-3]*0.98 and cost >= self.cost_at_depth[depth-4]*0.99 and cost >= self.cost_at_depth[depth-5]*0.993 and cost >= self.cost_at_depth[depth-6]*0.995 and cost >= self.cost_at_depth[depth-7]*0.998 and cost >= self.cost_at_depth[depth-8]*0.999 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 400 and (cost >= self.cost_at_depth[depth]*0.98 and cost >= self.cost_at_depth[depth-1]*0.99 and cost >= self.cost_at_depth[depth-2]*0.995 and cost >= self.cost_at_depth[depth-3]):
        #    pass
        #elif depth <= 450 and (cost >= self.cost_at_depth[depth]*0.99 and cost >= self.cost_at_depth[depth-1]):
        #    pass
        #elif cost >= self.cost_at_depth[depth]:
        #    pass
        #else:
        #    return
        
        elif depth <= 14:
            pass
        elif depth <= 30 and (cost >= self.cost_at_depth[depth]*0.82 and cost >= self.cost_at_depth[depth-1]*0.86 and cost >= self.cost_at_depth[depth-2]*0.9 and cost >= self.cost_at_depth[depth-3]*0.92 and cost >= self.cost_at_depth[depth-4]*0.94 and cost >= self.cost_at_depth[depth-5]*0.952 and cost >= self.cost_at_depth[depth-6]*0.96 and cost >= self.cost_at_depth[depth-7]*0.97 and cost >= self.cost_at_depth[depth-8]*0.98):
            pass
        elif depth <= 70 and (cost >= self.cost_at_depth[depth]*0.86 and cost >= self.cost_at_depth[depth-1]*0.9 and cost >= self.cost_at_depth[depth-2]*0.92 and cost >= self.cost_at_depth[depth-3]*0.94 and cost >= self.cost_at_depth[depth-4]*0.952 and cost >= self.cost_at_depth[depth-5]*0.96 and cost >= self.cost_at_depth[depth-6]*0.97 and cost >= self.cost_at_depth[depth-7]*0.98 and cost >= self.cost_at_depth[depth-8]*0.99):
            pass
        elif depth <= 100 and (cost >= self.cost_at_depth[depth]*0.92 and cost >= self.cost_at_depth[depth-1]*0.94 and cost >= self.cost_at_depth[depth-2]*0.952 and cost >= self.cost_at_depth[depth-3]*0.96 and cost >= self.cost_at_depth[depth-4]*0.97 and cost >= self.cost_at_depth[depth-5]*0.98 and cost >= self.cost_at_depth[depth-6]*0.99 and cost >= self.cost_at_depth[depth-7]*0.993 and cost >= self.cost_at_depth[depth-8]*0.995 and cost >= self.cost_at_depth[depth-9]):
            pass
        elif depth <= 150 and (cost >= self.cost_at_depth[depth]*0.93 and cost >= self.cost_at_depth[depth-1]*0.945 and cost >= self.cost_at_depth[depth-2]*0.96 and cost >= self.cost_at_depth[depth-3]*0.97 and cost >= self.cost_at_depth[depth-4]*0.98 and cost >= self.cost_at_depth[depth-5]*0.99 and cost >= self.cost_at_depth[depth-6]):
            pass
        elif depth <= 250 and (cost >= self.cost_at_depth[depth]*0.96 and cost >= self.cost_at_depth[depth-1]*0.98 and cost >= self.cost_at_depth[depth-2]*0.99 and cost >= self.cost_at_depth[depth-3]*0.995 and cost >= self.cost_at_depth[depth-4]):
            pass
        elif depth <= 400 and (cost >= self.cost_at_depth[depth]*0.995 and cost >= self.cost_at_depth[depth-1]):
            pass
        elif cost >= self.cost_at_depth[depth]:
            pass
        else:
            return
        #if cost > self.best_cost:
        #    return
        #Explore the children
        #best_solution_en_place(solution, self.dict)
        #manager = Manager()
        #solution = manager.dict(solution)
        #process = Process(target=best_solution_en_place, args=(solution,self.dict))
        #process.start()
        #process.join()

        child_list = []
        bid = random.random()
        if bid > 0.21:
            action = self.possible_action[0]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
            action = self.possible_action[1]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
        else:
            action = self.possible_action[1]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
            action = self.possible_action[0]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)


    def __mutate__(self, solution,action):
        child = copy.deepcopy(solution)
        child.step(action)
        return child

class QuickBranchAndBound:
    #Add penalty to invalid solutions
    def __init__(self, solution:Env,use_json=False):
        self.solution = solution
        self.max_depth = solution.max_step
        self.best_cost = 0
        self.best_sol = None
        self.possible_action = self.solution.getPossibleActionsAsInt()
        self.n_children = len(self.possible_action)
        self.cost_at_depth = dict()
        self.time_start = time.time()
        self.time_in = 10000
        if not use_json:
            for i in range(self.max_depth):
                self.cost_at_depth[i] = 0
        else:
            f = open('best_at_depth.json')
            data = json.load(f)
            for k,v in data.items():
                self.cost_at_depth[int(k)] = v
                


    def run(self):
        self.__branch_and_bound__(self.solution)
        return self.best_sol
        
    def __branch_and_bound__(self, solution:Env):
        if time.time()-self.time_start > self.time_in:
            return
        if solution.step_num == self.max_depth:
            return
        #Compute the cost of the solution
        cost = solution.score
        #If the solution is valid, update the best solution
        depth = solution.step_num

        if cost > self.best_cost:
            self.best_cost = cost
            self.best_sol = solution

        #If the cost is lower than the best solution, stop the exploration
        if (depth == 0 or cost >= self.cost_at_depth[depth-1]) and (self.cost_at_depth[depth] == 0 or cost > self.cost_at_depth[depth]):
            self.cost_at_depth[depth] = cost
            with open('best_at_depth.json','w+') as fp:
                json.dump(self.cost_at_depth,fp)
            print("Depth: ", depth, " - Cost: ", cost, " - Best cost: ", self.best_cost)
        
        
        #elif depth <= 15:
        #    pass
        #elif depth <= 50 and (cost >= self.cost_at_depth[depth]*0.8 and cost >= self.cost_at_depth[depth-1]*0.84 and cost >= self.cost_at_depth[depth-2]*0.86 and cost >= self.cost_at_depth[depth-3]*0.9 and cost >= self.cost_at_depth[depth-4]*0.92 and cost >= self.cost_at_depth[depth-5]*0.94 and cost >= self.cost_at_depth[depth-6]*0.952 and cost >= self.cost_at_depth[depth-7]*0.96 and cost >= self.cost_at_depth[depth-8]*0.97):
        #    pass
        #elif depth <= 80 and (cost >= self.cost_at_depth[depth]*0.86 and cost >= self.cost_at_depth[depth-1]*0.9 and cost >= self.cost_at_depth[depth-2]*0.92 and cost >= self.cost_at_depth[depth-3]*0.94 and cost >= self.cost_at_depth[depth-4]*0.952 and cost >= self.cost_at_depth[depth-5]*0.96 and cost >= self.cost_at_depth[depth-6]*0.97 and cost >= self.cost_at_depth[depth-7]*0.98 and cost >= self.cost_at_depth[depth-8]*0.99):
        #    pass
        #elif depth <= 100 and (cost >= self.cost_at_depth[depth]*0.92 and cost >= self.cost_at_depth[depth-1]*0.94 and cost >= self.cost_at_depth[depth-2]*0.952 and cost >= self.cost_at_depth[depth-3]*0.96 and cost >= self.cost_at_depth[depth-4]*0.97 and cost >= self.cost_at_depth[depth-5]*0.98 and cost >= self.cost_at_depth[depth-6]*0.99 and cost >= self.cost_at_depth[depth-7]*0.993 and cost >= self.cost_at_depth[depth-8]*0.995 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 200 and (cost >= self.cost_at_depth[depth]*0.94 and cost >= self.cost_at_depth[depth-1]*0.952 and cost >= self.cost_at_depth[depth-2]*0.96 and cost >= self.cost_at_depth[depth-3]*0.97 and cost >= self.cost_at_depth[depth-4]*0.98 and cost >= self.cost_at_depth[depth-5]*0.99 and cost >= self.cost_at_depth[depth-6]*0.993 and cost >= self.cost_at_depth[depth-7]*0.995 and cost >= self.cost_at_depth[depth-8]*0.998 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 300 and (cost >= self.cost_at_depth[depth]*0.95 and cost >= self.cost_at_depth[depth-1]*0.96 and cost >= self.cost_at_depth[depth-2]*0.97 and cost >= self.cost_at_depth[depth-3]*0.98 and cost >= self.cost_at_depth[depth-4]*0.99 and cost >= self.cost_at_depth[depth-5]*0.993 and cost >= self.cost_at_depth[depth-6]*0.995 and cost >= self.cost_at_depth[depth-7]*0.998 and cost >= self.cost_at_depth[depth-8]*0.999 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 400 and (cost >= self.cost_at_depth[depth]*0.98 and cost >= self.cost_at_depth[depth-1]*0.99 and cost >= self.cost_at_depth[depth-2]*0.995 and cost >= self.cost_at_depth[depth-3]):
        #    pass
        #elif depth <= 450 and (cost >= self.cost_at_depth[depth]*0.99 and cost >= self.cost_at_depth[depth-1]):
        #    pass
        #elif cost >= self.cost_at_depth[depth]:
        #    pass
        #else:
        #    return
        
        elif depth <= 14:
            pass
        elif depth <= 30 and (cost >= self.cost_at_depth[depth]*0.84 and cost >= self.cost_at_depth[depth-1]*0.86 and cost >= self.cost_at_depth[depth-2]*0.9 and cost >= self.cost_at_depth[depth-3]*0.92 and cost >= self.cost_at_depth[depth-4]*0.94 and cost >= self.cost_at_depth[depth-5]*0.952 and cost >= self.cost_at_depth[depth-6]*0.96 and cost >= self.cost_at_depth[depth-7]*0.97 and cost >= self.cost_at_depth[depth-8]*0.98):
            pass
        elif depth <= 70 and (cost >= self.cost_at_depth[depth]*0.9 and cost >= self.cost_at_depth[depth-1]*0.92 and cost >= self.cost_at_depth[depth-2]*0.94 and cost >= self.cost_at_depth[depth-3]*0.952 and cost >= self.cost_at_depth[depth-4]*0.96 and cost >= self.cost_at_depth[depth-5]*0.97 and cost >= self.cost_at_depth[depth-6]*0.98 and cost >= self.cost_at_depth[depth-7]*0.99 and cost >= self.cost_at_depth[depth-8]*0.993):
            pass
        elif depth <= 100 and (cost >= self.cost_at_depth[depth]*0.92 and cost >= self.cost_at_depth[depth-1]*0.94 and cost >= self.cost_at_depth[depth-2]*0.952 and cost >= self.cost_at_depth[depth-3]*0.96 and cost >= self.cost_at_depth[depth-4]*0.97 and cost >= self.cost_at_depth[depth-5]*0.98 and cost >= self.cost_at_depth[depth-6]*0.99 and cost >= self.cost_at_depth[depth-7]*0.993 and cost >= self.cost_at_depth[depth-8]*0.995 and cost >= self.cost_at_depth[depth-9]):
            pass
        elif depth <= 150 and (cost >= self.cost_at_depth[depth]*0.94 and cost >= self.cost_at_depth[depth-1]*0.953 and cost >= self.cost_at_depth[depth-2]*0.965 and cost >= self.cost_at_depth[depth-3]*0.973 and cost >= self.cost_at_depth[depth-4]*0.98 and cost >= self.cost_at_depth[depth-5]*0.99 and cost >= self.cost_at_depth[depth-6]):
            pass
        elif depth <= 250 and (cost >= self.cost_at_depth[depth]*0.96 and cost >= self.cost_at_depth[depth-1]*0.98 and cost >= self.cost_at_depth[depth-2]*0.99 and cost >= self.cost_at_depth[depth-3]*0.995 and cost >= self.cost_at_depth[depth-4]):
            pass
        elif depth <= 400 and (cost >= self.cost_at_depth[depth]*0.995 and cost >= self.cost_at_depth[depth-1]):
            pass
        elif cost >= self.cost_at_depth[depth]:
            pass
        else:
            return
        #if cost > self.best_cost:
        #    return
        #Explore the children
        #best_solution_en_place(solution, self.dict)
        #manager = Manager()
        #solution = manager.dict(solution)
        #process = Process(target=best_solution_en_place, args=(solution,self.dict))
        #process.start()
        #process.join()

        child_list = []
        bid = random.random()
        if bid > 0.21:
            action = self.possible_action[0]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
            action = self.possible_action[1]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
        else:
            action = self.possible_action[1]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
            action = self.possible_action[0]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)


    def __mutate__(self, solution,action):
        child = copy.deepcopy(solution)
        child.step(action)
        return child
    
class CVBranchAndBound:
    #Add penalty to invalid solutions
    def __init__(self, solution:Env,use_json=False):
        self.solution = solution
        self.max_depth = solution.max_step
        self.best_cost = 0
        self.best_sol = None
        self.possible_action = self.solution.getPossibleActionsAsInt()
        self.n_children = len(self.possible_action)
        self.cost_at_depth = dict()
        self.time_start = time.time()
        self.time_in = 10000
        if not use_json:
            for i in range(self.max_depth):
                self.cost_at_depth[i] = 0
        else:
            f = open('best_at_depth.json')
            data = json.load(f)
            for k,v in data.items():
                self.cost_at_depth[int(k)] = v
                


    def run(self):
        self.__branch_and_bound__(self.solution)
        return self.best_sol
        
    def __branch_and_bound__(self, solution:Env):
        if time.time()-self.time_start > self.time_in:
            return
        if solution.step_num == self.max_depth:
            return
        #Compute the cost of the solution
        cost = solution.score
        #If the solution is valid, update the best solution
        depth = solution.step_num

        if cost > self.best_cost:
            self.best_cost = cost
            self.best_sol = solution

        #If the cost is lower than the best solution, stop the exploration
        if (depth == 0 or cost >= self.cost_at_depth[depth-1]) and (self.cost_at_depth[depth] == 0 or cost > self.cost_at_depth[depth]):
            self.cost_at_depth[depth] = cost
            with open('best_at_depth.json','w+') as fp:
                json.dump(self.cost_at_depth,fp)
            print("Depth: ", depth, " - Cost: ", cost, " - Best cost: ", self.best_cost)
        
        
        #elif depth <= 15:
        #    pass
        #elif depth <= 50 and (cost >= self.cost_at_depth[depth]*0.8 and cost >= self.cost_at_depth[depth-1]*0.84 and cost >= self.cost_at_depth[depth-2]*0.86 and cost >= self.cost_at_depth[depth-3]*0.9 and cost >= self.cost_at_depth[depth-4]*0.92 and cost >= self.cost_at_depth[depth-5]*0.94 and cost >= self.cost_at_depth[depth-6]*0.952 and cost >= self.cost_at_depth[depth-7]*0.96 and cost >= self.cost_at_depth[depth-8]*0.97):
        #    pass
        #elif depth <= 80 and (cost >= self.cost_at_depth[depth]*0.86 and cost >= self.cost_at_depth[depth-1]*0.9 and cost >= self.cost_at_depth[depth-2]*0.92 and cost >= self.cost_at_depth[depth-3]*0.94 and cost >= self.cost_at_depth[depth-4]*0.952 and cost >= self.cost_at_depth[depth-5]*0.96 and cost >= self.cost_at_depth[depth-6]*0.97 and cost >= self.cost_at_depth[depth-7]*0.98 and cost >= self.cost_at_depth[depth-8]*0.99):
        #    pass
        #elif depth <= 100 and (cost >= self.cost_at_depth[depth]*0.92 and cost >= self.cost_at_depth[depth-1]*0.94 and cost >= self.cost_at_depth[depth-2]*0.952 and cost >= self.cost_at_depth[depth-3]*0.96 and cost >= self.cost_at_depth[depth-4]*0.97 and cost >= self.cost_at_depth[depth-5]*0.98 and cost >= self.cost_at_depth[depth-6]*0.99 and cost >= self.cost_at_depth[depth-7]*0.993 and cost >= self.cost_at_depth[depth-8]*0.995 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 200 and (cost >= self.cost_at_depth[depth]*0.94 and cost >= self.cost_at_depth[depth-1]*0.952 and cost >= self.cost_at_depth[depth-2]*0.96 and cost >= self.cost_at_depth[depth-3]*0.97 and cost >= self.cost_at_depth[depth-4]*0.98 and cost >= self.cost_at_depth[depth-5]*0.99 and cost >= self.cost_at_depth[depth-6]*0.993 and cost >= self.cost_at_depth[depth-7]*0.995 and cost >= self.cost_at_depth[depth-8]*0.998 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 300 and (cost >= self.cost_at_depth[depth]*0.95 and cost >= self.cost_at_depth[depth-1]*0.96 and cost >= self.cost_at_depth[depth-2]*0.97 and cost >= self.cost_at_depth[depth-3]*0.98 and cost >= self.cost_at_depth[depth-4]*0.99 and cost >= self.cost_at_depth[depth-5]*0.993 and cost >= self.cost_at_depth[depth-6]*0.995 and cost >= self.cost_at_depth[depth-7]*0.998 and cost >= self.cost_at_depth[depth-8]*0.999 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 400 and (cost >= self.cost_at_depth[depth]*0.98 and cost >= self.cost_at_depth[depth-1]*0.99 and cost >= self.cost_at_depth[depth-2]*0.995 and cost >= self.cost_at_depth[depth-3]):
        #    pass
        #elif depth <= 450 and (cost >= self.cost_at_depth[depth]*0.99 and cost >= self.cost_at_depth[depth-1]):
        #    pass
        #elif cost >= self.cost_at_depth[depth]:
        #    pass
        #else:
        #    return
        
        elif depth <= 14:
            pass
        elif depth <= 30 and (cost >= self.cost_at_depth[depth]*0.8 and cost >= self.cost_at_depth[depth-1]*0.82 and cost >= self.cost_at_depth[depth-2]*0.85 and cost >= self.cost_at_depth[depth-3]*0.87 and cost >= self.cost_at_depth[depth-4]*0.89 and cost >= self.cost_at_depth[depth-5]*0.9 and cost >= self.cost_at_depth[depth-6]*0.91 and cost >= self.cost_at_depth[depth-7]*0.92 and cost >= self.cost_at_depth[depth-8]*0.93):
            pass
        elif depth <= 120 and (cost >= self.cost_at_depth[depth]*0.85 and cost >= self.cost_at_depth[depth-1]*0.87 and cost >= self.cost_at_depth[depth-2]*0.89 and cost >= self.cost_at_depth[depth-3]*0.9 and cost >= self.cost_at_depth[depth-4]*0.92 and cost >= self.cost_at_depth[depth-5]*0.93 and cost >= self.cost_at_depth[depth-6]*0.95 and cost >= self.cost_at_depth[depth-7]*0.96 and cost >= self.cost_at_depth[depth-8]*0.97):
            pass
        elif depth <= 150 and (cost >= self.cost_at_depth[depth]*0.87 and cost >= self.cost_at_depth[depth-1]*0.9 and cost >= self.cost_at_depth[depth-2]*0.92 and cost >= self.cost_at_depth[depth-3]*0.94 and cost >= self.cost_at_depth[depth-4]*0.95 and cost >= self.cost_at_depth[depth-5]*0.96 and cost >= self.cost_at_depth[depth-6]*0.97 and cost >= self.cost_at_depth[depth-7]*0.98 and cost >= self.cost_at_depth[depth-8]*0.99 and cost >= self.cost_at_depth[depth-9]):
            pass
        elif depth <= 290 and (cost >= self.cost_at_depth[depth]*0.9 and cost >= self.cost_at_depth[depth-1]*0.92 and cost >= self.cost_at_depth[depth-2]*0.93 and cost >= self.cost_at_depth[depth-3]*0.95 and cost >= self.cost_at_depth[depth-4]*0.96 and cost >= self.cost_at_depth[depth-5]*0.97 and cost >= self.cost_at_depth[depth-6]*0.98 and cost >= self.cost_at_depth[depth-7]*0.99 and cost >= self.cost_at_depth[depth-8]*0.995 and cost >= self.cost_at_depth[depth-9]):
            pass
        elif depth <= 350 and (cost >= self.cost_at_depth[depth]*0.95 and cost >= self.cost_at_depth[depth-1]*0.96 and cost >= self.cost_at_depth[depth-2]*0.97 and cost >= self.cost_at_depth[depth-3]*0.98 and cost >= self.cost_at_depth[depth-4]*0.99 and cost >= self.cost_at_depth[depth-5]*0.995 and cost >= self.cost_at_depth[depth-6]):
            pass
        elif depth <= 430 and (cost >= self.cost_at_depth[depth]*0.98 and cost >= self.cost_at_depth[depth-1]*0.99 and cost >= self.cost_at_depth[depth-2]*0.995 and cost >= self.cost_at_depth[depth-3]):
            pass
        elif (cost >= self.cost_at_depth[depth]*0.995 and cost >= self.cost_at_depth[depth-1]):
            pass
        else:
            return
        #if cost > self.best_cost:
        #    return
        #Explore the children
        #best_solution_en_place(solution, self.dict)
        #manager = Manager()
        #solution = manager.dict(solution)
        #process = Process(target=best_solution_en_place, args=(solution,self.dict))
        #process.start()
        #process.join()

        child_list = []
        bid = random.random()
        if bid > 0.21:
            action = self.possible_action[0]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
            action = self.possible_action[1]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
        else:
            action = self.possible_action[1]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
            action = self.possible_action[0]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)


    def __mutate__(self, solution,action):
        child = copy.deepcopy(solution)
        child.step(action)
        return child

class MinusBranchAndBound:
    #Add penalty to invalid solutions
    def __init__(self, solution:Env,use_json=False):
        self.solution = solution
        self.max_depth = solution.max_step
        self.best_cost = 0
        self.best_sol = None
        self.possible_action = self.solution.getPossibleActionsAsInt()
        self.n_children = len(self.possible_action)
        self.cost_at_depth = dict()
        self.time_start = time.time()
        self.time_in = 1
        if not use_json:
            for i in range(self.max_depth):
                self.cost_at_depth[i] = 0
        else:
            f = open('best_at_depth.json')
            data = json.load(f)
            for k,v in data.items():
                self.cost_at_depth[int(k)] = v
                


    def run(self):
        self.__branch_and_bound__(self.solution)
        return self.best_sol
        
    def __branch_and_bound__(self, solution:Env):
        if time.time()-self.time_start > self.time_in:
            return
        if solution.state.step_num == self.max_depth:
            return
        #Compute the cost of the solution
        cost = solution.score
        #If the solution is valid, update the best solution
        depth = solution.state.step_num

        if cost > self.best_cost:
            self.best_cost = cost
            self.best_sol = solution
            print(cost)

        #If the cost is lower than the best solution, stop the exploration
        if (depth == 0 or cost >= self.cost_at_depth[depth-1]) and (self.cost_at_depth[depth] == 0 or cost > self.cost_at_depth[depth]):
            self.cost_at_depth[depth] = cost
            with open('best_at_depth.json','w+') as fp:
                json.dump(self.cost_at_depth,fp)
            print("Depth: ", depth, " - Cost: ", cost, " - Best cost: ", self.best_cost)
        
        
        #elif depth <= 15:
        #    pass
        #elif depth <= 50 and (cost >= self.cost_at_depth[depth]*0.8 and cost >= self.cost_at_depth[depth-1]*0.84 and cost >= self.cost_at_depth[depth-2]*0.86 and cost >= self.cost_at_depth[depth-3]*0.9 and cost >= self.cost_at_depth[depth-4]*0.92 and cost >= self.cost_at_depth[depth-5]*0.94 and cost >= self.cost_at_depth[depth-6]*0.952 and cost >= self.cost_at_depth[depth-7]*0.96 and cost >= self.cost_at_depth[depth-8]*0.97):
        #    pass
        #elif depth <= 80 and (cost >= self.cost_at_depth[depth]*0.86 and cost >= self.cost_at_depth[depth-1]*0.9 and cost >= self.cost_at_depth[depth-2]*0.92 and cost >= self.cost_at_depth[depth-3]*0.94 and cost >= self.cost_at_depth[depth-4]*0.952 and cost >= self.cost_at_depth[depth-5]*0.96 and cost >= self.cost_at_depth[depth-6]*0.97 and cost >= self.cost_at_depth[depth-7]*0.98 and cost >= self.cost_at_depth[depth-8]*0.99):
        #    pass
        #elif depth <= 100 and (cost >= self.cost_at_depth[depth]*0.92 and cost >= self.cost_at_depth[depth-1]*0.94 and cost >= self.cost_at_depth[depth-2]*0.952 and cost >= self.cost_at_depth[depth-3]*0.96 and cost >= self.cost_at_depth[depth-4]*0.97 and cost >= self.cost_at_depth[depth-5]*0.98 and cost >= self.cost_at_depth[depth-6]*0.99 and cost >= self.cost_at_depth[depth-7]*0.993 and cost >= self.cost_at_depth[depth-8]*0.995 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 200 and (cost >= self.cost_at_depth[depth]*0.94 and cost >= self.cost_at_depth[depth-1]*0.952 and cost >= self.cost_at_depth[depth-2]*0.96 and cost >= self.cost_at_depth[depth-3]*0.97 and cost >= self.cost_at_depth[depth-4]*0.98 and cost >= self.cost_at_depth[depth-5]*0.99 and cost >= self.cost_at_depth[depth-6]*0.993 and cost >= self.cost_at_depth[depth-7]*0.995 and cost >= self.cost_at_depth[depth-8]*0.998 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 300 and (cost >= self.cost_at_depth[depth]*0.95 and cost >= self.cost_at_depth[depth-1]*0.96 and cost >= self.cost_at_depth[depth-2]*0.97 and cost >= self.cost_at_depth[depth-3]*0.98 and cost >= self.cost_at_depth[depth-4]*0.99 and cost >= self.cost_at_depth[depth-5]*0.993 and cost >= self.cost_at_depth[depth-6]*0.995 and cost >= self.cost_at_depth[depth-7]*0.998 and cost >= self.cost_at_depth[depth-8]*0.999 and cost >= self.cost_at_depth[depth-9]):
        #    pass
        #elif depth <= 400 and (cost >= self.cost_at_depth[depth]*0.98 and cost >= self.cost_at_depth[depth-1]*0.99 and cost >= self.cost_at_depth[depth-2]*0.995 and cost >= self.cost_at_depth[depth-3]):
        #    pass
        #elif depth <= 450 and (cost >= self.cost_at_depth[depth]*0.99 and cost >= self.cost_at_depth[depth-1]):
        #    pass
        #elif cost >= self.cost_at_depth[depth]:
        #    pass
        #else:
        #    return
        
        #elif (cost >= self.cost_at_depth[depth]-2.41*depth-128-depth*depth*0.0032+depth*depth*depth*0.0000035+depth*depth*depth*depth*0.0000000115):
        elif (cost >= self.cost_at_depth[depth]-2.45*depth-120-depth*depth*0.0032+depth*depth*depth*0.0000075+depth*depth*depth*depth*0.000000018-depth*depth*depth*depth*depth*0.0000000000035):
            pass
        else:
            return
        #if cost > self.best_cost:
        #    return
        #Explore the children
        #best_solution_en_place(solution, self.dict)
        #manager = Manager()
        #solution = manager.dict(solution)
        #process = Process(target=best_solution_en_place, args=(solution,self.dict))
        #process.start()
        #process.join()

        child_list = []
        bid = random.random()
        if bid > 0.21:
            action = self.possible_action[0]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
            action = self.possible_action[1]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
        else:
            action = self.possible_action[1]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)
            action = self.possible_action[0]
            child = self.__mutate__(solution,action)
            self.__branch_and_bound__(child)


    def __mutate__(self, solution,action):
        child = copy.deepcopy(solution)
        child.step(action)
        return child

best_score = 0
timer = time.time()
while 1:
    
    env = Env(42)
    env.reset()
    algo = QuickBranchAndBound(env,use_json=False)
    best = algo.run()
    print("Best cost: ", best.score)
    if best.score >= best_score:
        best_score = best.score
        print("Best overall : ", best_score)
        if len(best.action_logs) == 499:
            # write the action log list into npy file
            np.save('perdictions.npy', np.array(best.action_logs))
        else:
            print(len(best.action_logs))
            raise Exception("Not 500")
    print(time.time()-timer)

# save in csv
