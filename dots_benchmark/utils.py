import numpy as np

def sample_init_points(f, fx, dims, n_samples:int=200):
    #200 initial points
    aaa = np.arange(f.lb[0], f.ub[0] + f.turn, f.turn).round(5)
    input_X = np.random.choice(aaa,size=(n_samples, dims))
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