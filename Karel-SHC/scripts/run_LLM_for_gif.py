import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
sys.dont_write_bytecode = True

sys.path.append('.')

from prog_policies.karel import KarelDSL
from prog_policies.karel_tasks import get_task_cls
from prog_policies.search_space import get_search_space_cls
from prog_policies.search_methods import get_search_method_cls
from prog_policies.search.utils import evaluate_program

# Django specific settings
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
import django
django.setup()

# Import your models for use in your script
from data.models import *

import uuid
import random
import numpy as np
import statistics

if __name__ == '__main__':
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    # parser.add_argument('--search_args_path', default='sample_args/search/latent_cem.json', help='Arguments path for search method')
    # parser.add_argument('--log_folder', default='logs', help='Folder to save logs')
    # parser.add_argument('--search_seed', type=int, help='Seed for search method')
    # parser.add_argument('--wandb_entity', type=str, help='Wandb entity')
    # parser.add_argument('--wandb_project', type=str, help='Wandb project')
    
    parser.add_argument('--search_space', default='ProgrammaticSpace', help='Search space class name')
    parser.add_argument('--search_method', default='HillClimbing', help='Search method class name')
    parser.add_argument('--seed', type=int, default=1, help='Random (integer) seed for searching')
    parser.add_argument('--num_iterations', type=int, default=1000000, help='Number of search iterations')
    parser.add_argument('--num_envs', type=int, default=32, help='Number of environments to search')
    parser.add_argument('--task', default='Harvester', help='Task class name')
    parser.add_argument('--sigma', type=float, default=0.1, help='Standard deviation for Gaussian noise in Latent Space')
    parser.add_argument('--k', type=int, default=32, help='Number of neighbors to consider')
    parser.add_argument('--e', type=int, default=8, help='Number of elite candidates in CEM-based methods')
    
    args = parser.parse_args()
    print('Mean for map ', args.task)
    dsl = KarelDSL()
    for d_number in range(1,2):
        dict_program = []
        number = random.randint(1,32)
        #path_dic = '/home/rubens/plots/KarelLLM/KarelLLMMoreInf/datasets/dictionaries/dicLLM/dict'+str(number)+'.txt'
        path_dic = '/home/rubens/pythonProjects/Karel-SHC/datasets/dictionaries/dicHarvesterCrashable/dict2.txt'
        print('Dictionary used: ',path_dic)             
        with open(path_dic) as f:
            for line in f.readlines():
                if ',' in line:
                    lines = line.split(',')                
                    dict_program.append(dsl.parse_str_to_node(lines[1].replace('"','').strip()))
                else:
                    dict_program.append(dsl.parse_str_to_node(line.replace('"','').strip()))
        print('Dictionary loaded!...')
        
        args.seed = random.randint(0,np.iinfo(np.int32).max)
        
        
        myuuid = uuid.uuid4()
        env_args = {
            "env_height": 8,
            "env_width": 8,
            "crashable": True,
            "leaps_behaviour": False,
            "max_calls": 10000
        }
        
        if args.task == "StairClimber" or args.task == "StairClimberSparse" or args.task == "TopOff" or args.task == "FourCorners" or args.task == 'MazeSparse':
            env_args["env_height"] = 12
            env_args["env_width"] = 12
        
        if args.task == "CleanHouse":
            env_args["env_height"] = 14
            env_args["env_width"] = 22
        
        task_cls = get_task_cls(args.task)
        task_envs = [task_cls(env_args, i) for i in range(args.num_envs)]
        
        search_space_cls = get_search_space_cls(args.search_space)
        search_space = search_space_cls(dsl, args.sigma)
        search_space.set_seed(args.seed)
        
        search_method_cls = get_search_method_cls(args.search_method)
        search_method = search_method_cls(args.k, args.e)
        
        best_reward = -float('inf')
        best_prog = None
        total_reward = []
        count = 0
        path = '/home/rubens/pythonProjects/recorded-semantic-search/Karel/Options_LISS/Harverster/'
        for prog in dict_program:            
            task_env = task_envs[0]            
            task_env.trace_program(prog,image_name=path+str(count)+'_K_'+args.task+'.gif')
            f = open(path+str(count)+'_K_'+args.task+'.txt', "w")
            f.write(dsl.parse_node_to_str(prog))
            f.close()
            count+=1
        
        
        #print(statistics.fmean(total_reward))