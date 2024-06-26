import os
import numpy as np
from .networks import SurrogateModelTraining 

from .functions import (
    Ackley,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Michalewicz,
    Griewank,
)

from .algorithms import (
    DualAnnealing,
    DifferentialEvolution,
    CMAES
)

FUNC = {'ackley':Ackley, 
        'rastrigin':Rastrigin, 
        'rosenbrock':Rosenbrock, 
        'schwefel':Schwefel, 
        'michalewicz':Michalewicz, 
        'griewank':Griewank
        }

DFO = {'da':DualAnnealing,
       'diff_evo':DifferentialEvolution,
       'cmaes':CMAES
       }

class DerivativeFreeOptimization:
    def __init__(self, 
                 dfo_method:str,
                 func:str,
                 dims:int,
                 num_samples:int,
                 surrogate:SurrogateModelTraining,
                 num_init_samples:int=200,
                 ):
        
        assert dims > 0
        assert num_samples > 0

        if func not in FUNC.keys():
            print('function not defined')
            os._exit(1)

        # initialisation
        self.num_samples = num_samples
        self.dfo_method = dfo_method
        self.dfo = DFO[dfo_method]
        self.dims = dims
        self.func = func
        self.f = FUNC[self.func](dims=self.dims, name=self.dfo_method+f'-{self.func}', iters = self.num_samples) 
        self.bounds = [(float(self.f.lb[i]), float(self.f.ub[i])) for i in range(len(self.f.lb))]
        self.surrogate = SurrogateModelTraining(f=self.func, dims=self.dims) if surrogate is None else surrogate
        self.rollout_round = 200 if (self.func == 'ackley') or (self.func == 'rastrigin') else 100
        self.num_init_samples = num_init_samples
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        # initialise samples (i.e. 200 data points)
        self.input_X, self.input_y2 = self.sampling_points(self.f, self.dims, self.num_init_samples)


    def sampling_points(self, f, dims:int=5, n_samples:int=200):
        dummy = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn).round(5)
        input_X = np.random.choice(dummy, size=(n_samples, dims))
        input_y = []
        input_y2 = []
        for i in input_X:
            y1, y2 = f(i)
            input_y.append(y1)
            input_y2.append(y2)
        input_X = np.array(input_X)
        input_y2 = np.array(input_y2)

        print("")
        print("="*20)
        print(f'{n_samples} initial data points collection completed, optimization started!')
        print("="*20)
        print("")

        return input_X, input_y2


    def run(self):
        if self.dfo_method == 'Random':
            out = self.sampling_points(self.f, dims=self.dims, n_samples=self.samples)
            for x in out['input_X']:
                init_y = self.f(x)
        else:
            for i in range(self.num_samples//20):
                model = self.surrogate(self.input_X, self.input_y2)
                optimizer = self.dfo(f=self.f, dims=self.dims, model=model, name=self.func)
                optimizer.mode = 'fast' # 'fast' or 'origin'
                print(f'This optimization is based on a ', optimizer.mode, ' mode {self.dfo_method} optimizer')
                top_X = optimizer.rollout(self.input_X, self.input_y2, self.rollout_round)
                top_y = []
                for xx in top_X:
                    _, y2 = self.f(xx)
                    top_y.append(y2)
                top_y = np.array(top_y)
                self.input_X=np.concatenate((self.input_X,top_X),axis=0)
                self.input_y2=np.concatenate((self.input_y2,top_y))
                
                if self.input_y2.min() == 0:
                    break

            