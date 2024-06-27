import numpy as np
from tensorflow import keras

def sampling_points(f, dims:int=5, n_samples:int=200):
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

    return {"input_X":input_X, "input_y2":input_y2}

class DivideLayer(keras.layers.Layer):
    def __init__(self, input_shape, const, **args):
        super().__init__()
        self.const = const
        self.w = self.add_weight(
            initializer='ones', shape=input_shape, trainable=False
        )

    def call(self, inputs):
        return keras.ops.divide(inputs, self.w) / self.const
