# import sys
# sys.path.append("..")

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import numpy as np

from ..networks import model_training 
from ..utils import sampling_points
from ..algorithms import (
    DualAnnealing,
    DifferentialEvolution,
    CMAES
)
from ..functions import (
    Ackley,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Michalewicz,
    Griewank,
)

FUNC = {'ackley':Ackley, 
        'rastrigin':Rastrigin, 
        'rosenbrock':Rosenbrock, 
        'schwefel':Schwefel, 
        'michalewicz':Michalewicz, 
        'griewank':Griewank
        }

parser = argparse.ArgumentParser(description='Process inputs')
parser.add_argument('--func', help='specify the test function')
parser.add_argument('--dims', type=int, help='specify the problem dimensions')
parser.add_argument('--samples', type=int, help='specify the number of samples to collect in the search')
parser.add_argument('--method', help='specify the method to search')
parser.add_argument('--init_samples', nargs='?', const=1, type=int, default=200)
args = parser.parse_args()


assert args.dims > 0
assert args.samples > 0

# Define function
f = None
if args.func in FUNC.keys():
    f = FUNC[args.func](dims=args.dims, name=args.method+f'-{args.func}', iters = args.samples)
else:
    print('function not defined')
    os._exit(1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

# Define surrogate model
nn = model_training(f=args.func, dims=args.dims)

# 200 initial points and set rollout_round
init_samples = sampling_points(f, dims=args.dims, n_samples=args.init_samples)
input_X, input_y2 = init_samples['input_X'], init_samples['input_y2']
rollout_round = 200 if args.func == 'ackley' or args.func == 'rastrigin' else 100

if args.method == 'Random':
    out = sampling_points(f, dims=args.dims, n_samples=args.samples)
    for x in out['input_X']:
        init_y = f(x)

elif args.method == 'DualAnnealing':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = DualAnnealing(f=f, dims=args.dims, model=model, name=args.func)
        optimizer.mode = 'fast' # 'fast' or 'origin'
        print('This optimization is based on a ', optimizer.mode, ' mode Dual Annealing optimizer')
        top_X = optimizer.rollout(input_X, input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = f(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if input_y2.min() == 0:
            break

elif args.method == 'DifferentialEvolution':
    for i in range(args.samples//20):
        model = nn(input_X, input_y2)
        optimizer = DifferentialEvolution(f=f, dims=args.dims, model=model, name=args.func)
        optimizer.mode = 'fast' # 'fast' or 'origin'
        print('This optimization is based on a ', optimizer.mode, ' mode Differential Evolution optimizer')
        top_X = optimizer.rollout(input_X, input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = f(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X, top_X),axis=0)
        input_y2=np.concatenate((input_y2, top_y))
        
        if input_y2.min() == 0:
            break

elif args.method == 'CMA-ES':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = CMAES(f = f, dims = args.dims, model=model, name = args.func)
        optimizer.mode = 'fast' # 'fast' or 'origin'
        print('This optimization is based on a ',optimizer.mode, ' mode CMA-ES optimizer')
        top_X = optimizer.rollout(input_X,input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = f(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if input_y2.min() == 0:
            break