from __future__ import annotations
from functools import partial
import copy

import torch

from prog_policies.base import dsl_nodes

from .simulated_annealing import SimulatedAnnealing
from .utils import evaluate_program

class LatentSimulatedAnnealing(SimulatedAnnealing):
    
    def parse_method_args(self, search_method_args: dict):
        super().parse_method_args(search_method_args)
        self.cem_sigma = search_method_args.get('cem_sigma', 0.25)
        self.cem_pop_size = search_method_args.get('cem_pop_size', 32)
        self.cem_elitism = search_method_args.get('cem_elitism', 0.1)
        self.cem_iterations = search_method_args.get('cem_iterations', 10)
        
    def random_program(self) -> dsl_nodes.Program:
        latent = torch.randn(1, self.latent_model.hidden_size, generator=self.torch_rng,
                             device=self.torch_device)
        program_tokens = self.latent_model.decode_vector(latent)[0]
        return self.dsl.parse_int_to_node(program_tokens)
    
    def find_and_cut(self, node: dsl_nodes.BaseNode, index_to_mutate: int) -> str:
        for i, child_type in enumerate(node.children_types):
            if index_to_mutate == self.current_index:
                if child_type == dsl_nodes.StatementNode:
                    current_child_str = self.dsl.parse_node_to_str(node.children[i])
                    node.children[i] = None
                    return current_child_str
                else:
                    return None
            else:
                self.current_index += 1
                return self.find_and_cut(node.children[i], index_to_mutate)
    
    def mutate_current_program(self) -> dsl_nodes.Program:
        copy_program = copy.deepcopy(self.current_program)
        
        found_valid_branch = False
        while not found_valid_branch:
            index = self.np_rng.randint(copy_program.get_size())
            self.current_index = 0
            branch_program_str = self.find_and_cut(copy_program, index)
            if branch_program_str is not None:
                found_valid_branch = True
        
        # At this point, `copy_program` has a hole in the branch `index`
        prog_with_hole = self.dsl.parse_node_to_str(copy_program)
        
        branch_program_tokens = self.dsl.parse_str_to_int(branch_program_str)
        branch_program_tokens = [self.dsl.t2i['DEF'], self.dsl.t2i['run'], self.dsl.t2i['m(']] + branch_program_tokens + [self.dsl.t2i['m)']]
        branch_program_tensor = torch.tensor(branch_program_tokens, dtype=torch.long, device=self.torch_device)
        starting_latent = self.latent_model.encode_program(branch_program_tensor).detach().repeat(self.cem_pop_size, 1)
        best_mutation_so_far = self.current_program
        best_reward_so_far = self.current_reward
        
        # CEM to fill the hole
        population = starting_latent + self.cem_sigma * torch.randn(self.cem_pop_size, self.latent_model.hidden_size,
                                                                    generator=self.torch_rng, device=self.torch_device)
        
        for cem_iter in range(1, self.cem_iterations + 1):
            population_tokens = self.latent_model.decode_vector(population)
            population_str = [self.dsl.parse_int_to_str(prog_tokens[3:-1]) for prog_tokens in population_tokens]
            
            full_programs_str = [prog_with_hole.replace('<HOLE>', prog_str) for prog_str in population_str]
            
            # num_unique_programs = len(set(full_programs_str))
            # self.log(f'CEM iteration {cem_iter}: {num_unique_programs} unique programs')
            
            if self.pool is not None:
                fn = partial(evaluate_program, dsl=self.dsl, task_envs=self.task_envs)
                rewards = self.pool.map(fn, full_programs_str)
            else:
                rewards = [evaluate_program(p, self.dsl, self.task_envs) for p in full_programs_str]
            
            for r, prog_str in zip(rewards, full_programs_str):
                self.num_evaluations += 1
                if r > best_reward_so_far:
                    best_mutation_so_far = self.dsl.parse_str_to_node(prog_str)
                    best_reward_so_far = r
                if r > self.best_reward:
                    self.best_reward = r
                    self.best_program = prog_str
                    self.save_best()
                    
                if self.best_reward >= 1.0:
                    return self.dsl.parse_str_to_node(prog_str)
                
            n_elite = int(self.cem_elitism * self.cem_pop_size)
            best_indices = torch.topk(torch.tensor(rewards, device=self.torch_device), n_elite).indices
            elite_population = population[best_indices]
            
            new_indices = torch.ones(elite_population.size(0), device=self.torch_device).multinomial(
                self.cem_pop_size, generator=self.torch_rng, replacement=True)
            new_population = []
            for index in new_indices:
                sample = elite_population[index]
                new_population.append(
                    sample + self.cem_sigma * torch.randn(self.latent_model.hidden_size,
                                                          generator=self.torch_rng,
                                                          device=self.torch_device)
                )
            population = torch.stack(new_population)
        
        return best_mutation_so_far
