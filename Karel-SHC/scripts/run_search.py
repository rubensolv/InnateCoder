import json
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import random
import numpy as np

sys.path.append('.')

from prog_policies.karel import KarelDSL, KarelEnvironment
from prog_policies.latent_space.models import *
from prog_policies.search import get_search_cls

if __name__ == '__main__':
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--search_args_path', default='sample_args/search/shc.json', help='Arguments path for search method')
    parser.add_argument('--log_folder', default='logs', help='Folder to save logs')
    parser.add_argument('--search_seed', default=2, type=int, help='Seed for search method')
    parser.add_argument('--wandb_entity', type=str, help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, help='Wandb project')
    parser.add_argument('--task', default='Harvester', help='Task class name')
    parser.add_argument('--qtd_dicts', default=-1,type=int, help='Quantity of dictionaries to be loaded in the test')
    
    args = parser.parse_args()
    # args.search_seed = random.randint(0,np.iinfo(np.int32).max)
    #args.search_seed = 1
    
    
    with open(args.search_args_path, 'r') as f:
        search_args = json.load(f)

        
    if args.qtd_dicts is not None:
        search_args['search_method_args']['qtd_dicts'] = args.qtd_dicts    
        
    if args.search_seed is not None:
        search_args['search_method_args']['search_seed'] = args.search_seed

    
    
    search_args['search_method_args']['task_cls_name'] = args.task
    
    if args.task == "StairClimber" or args.task == "StairClimberSparse" or args.task == "TopOff" or args.task == "FourCorners" or args.task == 'MazeSparse':
        search_args['search_method_args']['env_args']["env_height"] = 12
        search_args['search_method_args']['env_args']["env_width"] = 12
    elif args.task == "CleanHouse":
        search_args['search_method_args']['env_args']["env_height"] = 14
        search_args['search_method_args']['env_args']["env_width"] = 22
    
    if args.wandb_project:
        wandb_args = {
            'project': args.wandb_project,
            'entity': args.wandb_entity,
        }
    else:
        wandb_args = None
    
    device = torch.device('cpu')
    
    dsl = KarelDSL()
    
    search_cls = get_search_cls(search_args["search_method_cls_name"])
    
    searcher = search_cls(
        dsl,
        KarelEnvironment,
        device,
        log_folder=args.log_folder,
        wandb_args=wandb_args,
        **search_args["search_method_args"]
    )
    
    searcher.search()
