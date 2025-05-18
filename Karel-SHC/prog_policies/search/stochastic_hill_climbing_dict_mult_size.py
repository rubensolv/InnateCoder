from __future__ import annotations
import copy

import numpy as np
import random
from os import listdir
from os.path import isfile, join

from prog_policies.base import dsl_nodes
from prog_policies.karel import KarelDSL, KarelEnvironment, KarelStateGenerator
from prog_policies.search_space import BaseSearchSpace, ProgrammaticSpace

from .base_search import BaseSearch
from .utils import evaluate_program


# recursively calculate the node depth (number of levels from root)
def get_max_depth(program: dsl_nodes.Program) -> int:
    depth = 0
    for child in program.children:
        if child is not None:
            depth = max(depth, get_max_depth(child))
    return depth + program.node_depth
    
# recursively calculate the max number of Concatenate nodes in a row
def get_max_sequence(program: dsl_nodes.Program, current_sequence = 1, max_sequence = 0) -> int:
    if isinstance(program, dsl_nodes.Concatenate):
        current_sequence += 1
    else:
        current_sequence = 1
    max_sequence = max(max_sequence, current_sequence)
    for child in program.children:
        max_sequence = max(max_sequence, get_max_sequence(child, current_sequence, max_sequence))
    return max_sequence

class DictionaryControl:
    def __init__(self, dsl, seed, task, qtd_dicts):
        self.dsl = dsl
        self.qtd_sizes_to_load = qtd_dicts
        self.task = task
        #self.path_dic = 'datasets/dictionaries/randomDicTest'+str(seed)+'.txt'
        self.path_dic = 'datasets/dictionaries/testDicts/'        
        self.dict_program = []
        self.env_generators = []
        self.dict_nodes = set()
        for i in range(32):
            size = random.randint(6, 12)
            env_args = {
                "env_height": size,
                "env_width": size,
                "crashable": False,
                "leaps_behaviour": True,
                "max_calls": 100
            }                    
            self.env_generators.append(KarelStateGenerator(env_args, i))
        self.initial_states = [env_generator.random_state() for env_generator in self.env_generators]
        self.load_dictionary()        
        
    def get_all_files_by_task(self):
        dicret = dict()
        onlyfiles = [f for f in listdir(self.path_dic) if isfile(join(self.path_dic, f))]
        for file in onlyfiles:
            frags = file.split('_')
            data = set()
            if frags[0] != self.task:
                if frags[0] in dicret:
                    #include in the list
                    data = dicret[frags[0]]                               
                data.add(file)
                dicret[frags[0]] = data        
        return dicret
            
    def get_files_to_load(self,dic_files):
        selected = set()
        while len(selected) < self.qtd_sizes_to_load:
            keys = list(dic_files.keys())
            id = random.randint(0,len(keys)-1)
            data = list(dic_files[keys[id]])
            item = data[random.randint(0,len(data)-1)]
            selected.add(item)
        return selected
    
    def load_dictionary(self)-> None:                                                                                                                                                                                                                                                                                                             
        print('Dictionaries path: ',self.path_dic)        
        #coletar os arquivos por nome da task em um dic.    
        dic_files = self.get_all_files_by_task()
        dics_to_load = self.get_files_to_load(dic_files)
        print('Dictionaries selected: ', dics_to_load)     
        # loading the dicts.  
        for dict in dics_to_load:
            print('Loading dict ', dict)
            with open(self.path_dic+dict) as f:
                for line in f.readlines():
                    if ',' in line:
                        lines = line.split(',')                
                        self.dict_program.append(self.dsl.parse_str_to_node(lines[1].replace('"','').strip()))
                    else:
                        self.dict_program.append(self.dsl.parse_str_to_node(line.replace('"','').strip()))
        print('Dicts loaded ')
    
    def check_trajectory_in_pool(self, trajectory, dic_pool) -> bool:
        for traj in dic_pool:
            sim = self.trajectories_similarity(trajectory, traj)
            if sim == 1:
                return True        
        return False
    
    def clean_dict_by_behavior(self)-> None:                        
        for s in self.initial_states:            
            traj_program = dict()
            for program in self.dict_program:                        
                trajectory=  self.get_trajectory(program, s)                                
                #check trajectory in pool
                if not self.check_trajectory_in_pool(trajectory, list(traj_program.values())):
                    #print(trajectories)            
                    traj_program[program] = trajectory
            self.dict_program = list(traj_program.keys())
        #print(len(self.dict_program))        
        self.compose_dict_nodes()    
    
    def get_trajectory(self, program: dsl_nodes.Program, initial_state: KarelEnvironment) -> list[dsl_nodes.Action]:
        tau = []
        for action in program.run_generator(copy.deepcopy(initial_state)):
            tau.append(action)
        return tau

    def trajectories_similarity(self, tau_a: list[dsl_nodes.Action], tau_b: list[dsl_nodes.Action]) -> float:
        similarity = 0.
        t_max = max(len(tau_a), len(tau_b))
        if t_max == 0: return 1.
        for i in range(min(len(tau_a), len(tau_b))):
            if tau_a[i].name == tau_b[i].name:
                similarity += 1.
            else:
                break
        similarity /= t_max
        return similarity

    def compose_dict_nodes(self) -> None:
        self.dict_nodes.clear()
        for t_progam in self.dict_program:
            self.dict_nodes.update(t_progam.get_all_nodes()[1:])
        # print(len(self.dict_nodes))
        # print()
            
        

class StochasticHillClimbingDictMultSize(BaseSearch):

    def parse_method_args(self, search_method_args: dict):
        self.k = search_method_args.get('k', 250)                       
        
    def init_search_vars(self):
        self.current_program = self.random_program()
        self.current_reward = evaluate_program(self.current_program, self.dsl, self.task_envs)
        self.num_evaluations = 1
        self.dicProgram = DictionaryControl(self.dsl, self.search_seed, self.task_cls_name, self.qtd_dicts)
        self.dicProgram.clean_dict_by_behavior()
        
    def get_search_vars(self) -> dict:
        return {
            'current_program': self.current_program,
            'current_reward': self.current_reward
        }
    
    def set_search_vars(self, search_vars: dict):
        self.current_program = search_vars.get('current_program')
        self.current_reward = search_vars.get('current_reward')
        
    def fill_children_of_node(self, node: dsl_nodes.BaseNode,
                          current_depth: int = 1, current_sequence: int = 0,
                          max_depth: int = 4, max_sequence: int = 6) -> None:
        node_prod_rules = self.dsl.prod_rules[type(node)]
        for i, child_type in enumerate(node.get_children_types()):
            child_probs = self.dsl.get_dsl_nodes_probs(child_type)
            for child_type in child_probs:
                if child_type not in node_prod_rules[i]:
                    child_probs[child_type] = 0.
                if current_depth >= max_depth and child_type.get_node_depth() > 0:
                    child_probs[child_type] = 0.
            if issubclass(type(node), dsl_nodes.Concatenate) and current_sequence + 1 >= max_sequence:
                if dsl_nodes.Concatenate in child_probs:
                    child_probs[dsl_nodes.Concatenate] = 0.
            
            p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
            child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
            child_instance = child()
            if child.get_number_children() > 0:
                if issubclass(type(node), dsl_nodes.Concatenate):
                    self.fill_children_of_node(child_instance, current_depth + child.get_node_depth(),
                                               current_sequence + 1, max_depth, max_sequence)
                else:
                    self.fill_children_of_node(child_instance, current_depth + child.get_node_depth(),
                                               1, max_depth, max_sequence)
            
            elif isinstance(child_instance, dsl_nodes.Action):
                child_instance.name = self.np_rng.choice(list(self.dsl.action_probs.keys()),
                                                         p=list(self.dsl.action_probs.values()))
            elif isinstance(child_instance, dsl_nodes.BoolFeature):
                child_instance.name = self.np_rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                         p=list(self.dsl.bool_feat_probs.values()))
            elif isinstance(child_instance, dsl_nodes.ConstInt):
                child_instance.value = self.np_rng.choice(list(self.dsl.const_int_probs.keys()),
                                                          p=list(self.dsl.const_int_probs.values()))
            node.children[i] = child_instance
            child_instance.parent = node
    
    def random_program(self) -> dsl_nodes.Program:
        program = dsl_nodes.Program()
        self.fill_children_of_node(program, max_depth=4, max_sequence=6)
        return program
    
    def mutate_node_by_dic(self, node_to_mutate: dsl_nodes.BaseNode) -> None:        
        for i, child in enumerate(node_to_mutate.parent.children):
            if child == node_to_mutate:
                child_type = node_to_mutate.parent.children_types[i]
                list_of_nodes = list(self.dicProgram.dict_nodes)
                random.shuffle(list_of_nodes)
                for node in list_of_nodes:
                    if isinstance(node, child_type):
                        child_instance = copy.deepcopy(node)
                        node_to_mutate.parent.children[i] = child_instance
                        child_instance.parent = node_to_mutate.parent
                        return
                
    def mutate_node(self, node_to_mutate: dsl_nodes.BaseNode) -> None:
        for i, child in enumerate(node_to_mutate.parent.children):
            if child == node_to_mutate:
                child_type = node_to_mutate.parent.children_types[i]
                node_prod_rules = self.dsl.prod_rules[type(node_to_mutate.parent)]
                child_probs = self.dsl.get_dsl_nodes_probs(child_type)
                for child_type in child_probs:
                    if child_type not in node_prod_rules[i]:
                        child_probs[child_type] = 0.
                
                p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
                child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
                child_instance = child()
                if child.get_number_children() > 0:
                    self.fill_children_of_node(child_instance, max_depth=2, max_sequence=4)
                elif isinstance(child_instance, dsl_nodes.Action):
                    child_instance.name = self.np_rng.choice(list(self.dsl.action_probs.keys()),
                                                                p=list(self.dsl.action_probs.values()))
                elif isinstance(child_instance, dsl_nodes.BoolFeature):
                    child_instance.name = self.np_rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                                p=list(self.dsl.bool_feat_probs.values()))
                elif isinstance(child_instance, dsl_nodes.ConstInt):
                    child_instance.value = self.np_rng.choice(list(self.dsl.const_int_probs.keys()),
                                                                p=list(self.dsl.const_int_probs.values()))
                node_to_mutate.parent.children[i] = child_instance
                child_instance.parent = node_to_mutate.parent
    
    def mutate_current_program(self) -> dsl_nodes.Program:
        accepted = False
        while not accepted:
            mutated_program = copy.deepcopy(self.current_program)
        
            node_to_mutate = self.np_rng.choice(mutated_program.get_all_nodes()[1:])
            #self.mutate_node(node_to_mutate)
            self.mutate_node_by_dic(node_to_mutate)
            
            if mutated_program.get_size() <= 20:
                accepted = True
        
        return mutated_program
    
    def search_iteration(self):
        if self.current_iteration % 100 == 0:
            self.log(f'Iteration {self.current_iteration}: Best reward {self.best_reward}, evaluations {self.num_evaluations}')
        
        if self.current_reward > self.best_reward:
            self.best_reward = self.current_reward
            self.best_program = self.dsl.parse_node_to_str(self.current_program)
            self.save_best()
        if self.best_reward >= 1.0:
            return
        
        neighbors = []
        neighbors_text = []
        for _ in range(self.k):
            accepted = False
            while not accepted:
                mutated_program = copy.deepcopy(self.current_program)                
                node_to_mutate = self.np_rng.choice(mutated_program.get_all_nodes()[1:])
                if random.uniform(0, 1) <= 0.2:
                    self.mutate_node(node_to_mutate)
                else:
                    self.mutate_node_by_dic(node_to_mutate)
                prog_str = self.dsl.parse_node_to_str(mutated_program)                
                accepted = get_max_depth(mutated_program) <= 4 and get_max_sequence(mutated_program) <= 6 and len(prog_str.split(" ")) <= 45
            if prog_str not in neighbors_text:
                neighbors.append(mutated_program)
                neighbors_text.append(prog_str)
                #print(prog_str)
        
        in_local_maximum = True
        for prog in neighbors:
            reward = evaluate_program(prog, self.dsl, self.task_envs)
            self.num_evaluations += 1
            if reward > self.current_reward:
                self.current_program = prog
                self.current_reward = reward
                in_local_maximum = False
                self.dicProgram.dict_program.append(copy.deepcopy(prog))                
                break
        self.dicProgram.compose_dict_nodes()
        if in_local_maximum:
            self.current_program = self.random_program()
            self.current_reward = evaluate_program(self.current_program, self.dsl, self.task_envs)
            self.num_evaluations += 1
