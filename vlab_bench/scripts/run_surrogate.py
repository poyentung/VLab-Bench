import os
import argparse
import numpy as np
from dots_benchmark.networks import model_training 
from dots_benchmark.algorithms import (
    DOTS,
    MCTS_Greedy,
    MCTS_eGreedy,
    DualAnnealing,
    DifferentialEvolution,
    CMAES
)
from dots_benchmark.functions import (
    Surrogate,
    Ackley,
    Rastrigin,
    Rosenbrock,
    Levy,
    Schwefel,
    Michalewicz,
    Griewank,
)

FUNC = {'ackley':Ackley, 
        'rastrigin':Rastrigin, 
        'rosenbrock':Rosenbrock, 
        'levy':Levy, 
        'schwefel':Schwefel, 
        'michalewicz':Michalewicz, 
        'griewank':Griewank
        }

parser = argparse.ArgumentParser(description='Process inputs')
parser.add_argument('--func', help='specify the test function')
parser.add_argument('--dims', type=int, help='specify the problem dimensions')
parser.add_argument('--samples', type=int, help='specify the number of samples to collect in the search')
parser.add_argument('--method', help='specify the method to search')
args = parser.parse_args()


assert args.dims > 0
assert args.samples > 0

# Define function
f = None
if args.func not in FUNC.keys():
    print('function not defined')
    os._exit(1)
else:
    f = FUNC[args.func](dims=args.dims)
    fx = Surrogate(dims=args.dims, name=args.method+f'-{args.func}', f=f, iters = args.samples)


nn = model_training(f=args.func, dims=args.dims)

assert f is not None
bounds = []
for idx in range(0, len(f.lb) ):
    bounds.append( ( float(f.lb[idx]), float(f.ub[idx])) )


#200 initial points
aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn).round(5)
input_X = np.random.choice(aaa,size=(200, args.dims))
input_y = []
input_y2 = []
for i in input_X:
    y1, y2 = fx(i)
    input_y.append(y1)
    input_y2.append(y2)
input_X = np.array(input_X)
input_y2 = np.array(input_y2)
print("")
print("="*20)
print("200 initial data points collection completed, optimization started!")
print("="*20)
print("")

if args.func == 'ackley' or args.func == 'rastrigin':
    rollout_round = 200
else:
    rollout_round = 100

if args.method == 'DOTS':
    if args.func == 'ackley':
        ratio = 0.1
    elif args.func == 'rastrigin':
        ratio = 1
    elif args.func == 'rosenbrock':
        ratio = 1
    else:
        ratio = 1
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = DOTS(f=f, model=model, name = args.func)
        top_X = optimizer.rollout(input_X,input_y2,rollout_round,ratio,i)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)])[0] == 0:
            break

elif args.method == 'DOTS-Greedy':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = MCTS_Greedy(f = f, dims = args.dims, model=model, name = args.func)
        top_X = optimizer.rollout(input_X,input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)]) == 0:
            break

elif args.method == 'DOTS-eGreedy':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = MCTS_eGreedy(f = f, dims = args.dims, model=model, name = args.func)
        top_X = optimizer.rollout(input_X,input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)]) == 0:
            break

elif args.method == 'Random':
    aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn)
    init_X = np.random.choice(aaa,size=(args.samples,args.dims))
    for i in init_X:
        init_y = fx(i)


elif args.method == 'DualAnnealing':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = DualAnnealing(f = f, dims = args.dims, model=model, name = args.func)
        optimizer.mode = 'fast' # 'fast' or 'origin'
        print('This optimization is based on a ',optimizer.mode, ' mode Dual Annealing optimizer')
        top_X = optimizer.rollout(input_X,input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)]) == 0:
            break

elif args.method == 'DifferentialEvolution':
    for i in range(args.samples//20):
        model = nn(input_X,input_y2)
        optimizer = DifferentialEvolution(f = f, dims = args.dims, model=model, name = args.func)
        optimizer.mode = 'fast' # 'fast' or 'origin'
        print('This optimization is based on a ',optimizer.mode, ' mode Differential Evolution optimizer')
        top_X = optimizer.rollout(input_X,input_y2, rollout_round)
        top_y = []
        for xx in top_X:
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)]) == 0:
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
            y1, y2 = fx(xx)
            top_y.append(y2)
        top_y = np.array(top_y)
        input_X=np.concatenate((input_X,top_X),axis=0)
        input_y2=np.concatenate((input_y2,top_y))
        
        if f(input_X[np.argmax(input_y2)]) == 0:
            break