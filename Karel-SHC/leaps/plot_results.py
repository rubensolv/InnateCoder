import os, glob, pickle
import numpy as np
from matplotlib import pyplot as plt

def load_files(task, model, seed):
    
    filepaths = [
        glob.glob(os.path.join(
            'results', task, f'{model}-handwritten-{seed}-*', f
        ))[0] for f in ['best_exec_data.pkl', 'best_program.txt', 'scores.pkl']
    ]
    
    [best_exec_data_pkl_path, best_program_txt_path, scores_pkl_path] = filepaths
    
    with open(best_exec_data_pkl_path, 'rb') as f:
        best_exec_data = pickle.load(f)
    
    with open(best_program_txt_path, 'r') as f:
        best_program = f.read()
    
    with open(scores_pkl_path, 'rb') as f:
        scores = pickle.load(f)
        
    scores = np.array(scores).flatten()
        
    return best_exec_data, best_program, scores

if __name__ == '__main__':
    
    model_prefix = 'LEAPSPL'
    
    tasks = ['leaps_maze', 'leaps_stairclimber', 'leaps_topoff', 'leaps_harvester', 'leaps_fourcorners', 'leaps_cleanhouse']
    
    os.makedirs('plots', exist_ok=True)
    
    model_sizes = [8, 16, 32, 64, 128, 256]
    mean_scores = []
    percentage_perfect_scores = []
    mean_iterations = []
    
    for task_name in tasks:
        
        task_mean_scores = []
        task_percentage_perfect_scores = []
        task_mean_iterations = []
    
        for model_size in model_sizes:
        
            scores_maxes = []
            num_iterations = []
            perfect_scores = []
            
            for seed in range(1, 21):
                best_exec_data, best_program, scores = load_files(task_name, f'{model_prefix}_{model_size}', seed)
                scores_maxes.append(scores.max())
                perfect_scores.append(np.isclose(scores.max(), 1.1))
                num_iterations.append(scores.size)
            
            task_mean_scores.append(scores_maxes)
            task_percentage_perfect_scores.append(np.sum(perfect_scores) / 20)
            task_mean_iterations.append(np.mean(num_iterations))

        mean_scores.append(task_mean_scores)
        percentage_perfect_scores.append(task_percentage_perfect_scores)
        mean_iterations.append(task_mean_iterations)
    
    plt.figure()
    for percentages in percentage_perfect_scores:
        plt.plot(model_sizes, percentages)
    plt.xticks(model_sizes)
    plt.title('Convergence ratio over 20 seeds (higher is better)')
    plt.xlabel('Model size (number of dimensions in latent space)')
    plt.ylabel('Percentage of convergence (program solves the task)')
    plt.grid()
    plt.legend(tasks)
    plt.savefig(os.path.join('plots', 'score_vs_modelsize.png'))
    
    plt.figure()
    for iterations in mean_iterations:
        plt.plot(model_sizes, iterations)
    plt.xticks(model_sizes)
    plt.title('Average of iterations over 20 seeds (lower is better)')
    plt.xlabel('Model size (number of dimensions in latent space)')
    plt.ylabel('Average of iterations until convergence')
    plt.grid()
    plt.legend(tasks)
    plt.savefig(os.path.join('plots', 'iterations_vs_modelsize.png'))
    
    pass