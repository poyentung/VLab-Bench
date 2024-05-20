# VLab-Bench

`VLab-Bench` is a suite that offers benchmarks for real-world scientific design tasks and optimisation algorithms for materials science and biology. 

## Current results
<style scoped>
table {
  font-size: 12px;
}
</style>
|  | Ackley-20 | Ackley-100 | Ackley-200 | Rastrigin-20 | Rastrigin-100 | Rastrigin-1000 | Rosenbrock-20 | Rosenbrock-60 | Rosenbrock-100 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **DOTS** | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 3.18 ± 5.18 | 11.10 ± 22.10 | 19.50 ± 39.10 |
| **TurBo5** | 0.37 ± 0.14 | 1.73 ± 0.18 | 4.88 ± 0.1 | 51.6 ± 3.7 | 400 ± 34 | 10,371 ± 136 | 27.4 ± 3.7 | 184 ± 66 | 1,268 ± 666 |
| **LaMCTS** | 1.96 ± 0.75 | 5.05 ± 0.73 | 7.95 ± 0.99 | 80 ± 30.3 | 817 ± 44 | 15,534 ± 136 | 84.8 ± 52.3 | 2,182 ± 953 | 6,517 ± 977 |
| **CMS-ES** | 0.75 ± 0.09 | 2.85 ± 0.04 | 3.50 ± 0.06 | 77.6 ± 3.2 | 974 ± 17 | 13530 ± 175 | 55.7 ± 35.4 | 148 ±31  | 365 ± 42 |
| **Diff-Evo** | 6.43 ± 0.16 | 8.13 ±  0.19| 8.50 ± 0.07 | 188 ± 12 | 1,299 ± 32 | 14,364 ± 193 | 7.97E3 ± 1.15E3 | 1.34E5 ± 1.11E4 | 2.83E5 ± 2.69E4 |
| **DA** | 0.0 ± 0.0 | 3.28 ± 0.19 | 6.77 ± 0.15 | 12.9 ± 6.3 | 527 ± 39 | 14,503 ± 148 | 49.2 ± 28.7 | 670 ± 76 | 9,082 ± 877 |
| **Shiwa** | 4.43 ± 0.07 | 5.78 ± 0.52 | 7.48 ± 0.20 | 248 ± 2 | 1192 ± 47 | 15,921 ± 204 | 22,659 ± 1,462 | 661 ± 111 | 2,408 ± 218 |
| **MCMC** | 0.0 ± 0.0 | 4.79 ± 0.16 | 7.82 ± 0.19 | 89.0 ± 27.2 | 727 ± 38 | 14,107 ± 221 | 109 ± 63 | 187 ± 26 | 884 ± 359 |
| **DOO** | 7.17 ± 0.37 | 9.44 ± 0.09 | 9.85 ± 0.05 | 222 ± 14 | 1,503 ± 44 | 17,335 ± 50 | 1.64E4 ± 4.56E3 | 3.42E5 ± 1.90E4 | 7.22E5 ± 2.70E4 |
| **SOO** | 7.75 ± 0.18 | 9.40 ± 0.17 | 9.80 ± 0.05 | 224 ± 8 | 1,537 ± 27 | 17,285 ± 73 | 2.76E4 ± 7.44E3 | 3.58E5 ± 2.4E4 | 7.63E5 ± 2.70E4 |
| **VOO** | 2.44 ± 0.49 | 5.23 ± 0.17 | 5.63 ± 0.03 | 103 ± 13 | 923 ± 28 | 10,975 ± 77 | 57.6 ± 8.4 | 3,331 ± 403 | 21,065 ± 3241 |
| **Random** | 7.59 ± 0.17 | 9.23 ± 0.13 | 9.61 ± 0.06 | 218 ± 15 | 1,473 ± 16 | 17,112 ± 98 | 2.38E4 ± 1.19E3 | 2.74E5 ± 2.01E4 | 6.46E5 ± 9.36E3 |

## Available real-world tasks

The [currently available tasks](vlab_bench/functions.py) are:

* Cyclic peptite binder design
* Electron ptychography: reconstruction optimisation

Please send us a PR to add your real-world task!

## Available synthetic function tasks

The [currently available functions](vlab_bench/functions.py) are:

* Ackley
* Rastrigin
* Rosenbrock
* Levy
* Schwefel
* Michalewicz
* Griewank

## Available optimisation algorithms

The [currently available algorithms](vlab_bench/algorithms.py) are:

* DOTS (Derivative-free stOchastic Tree Search, [Wei et al., 2024](https://arxiv.org/abs/2404.04062))
* MCTS_Greedy
* MCTS_eGreedy
* DA (Dual Annealing, default setting in [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#rbaa258a99356-5))
* Diff-Evo (Differential Evolution, default setting in [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html))
* CMA-ES (Differential Evolution Default in Scipy)

Please send us a PR to add your algorithm!

## Installation

The code requires `python>=3.9`. Installation Tensorflow and Keras with CUDA support is stroongly recommended.

Install DOTS:

```
pip install git+https://github.com/poyentung/vlab_bench.git
```

or clone the repository to local devices:

```
git clone git@github.com:poyentung/vlab_bench.git
cd DOTS; pip install -e .
```

## Quick start

Here we evaluate DOTS on Ackley in 10 dimensions for 1000 samples.

- **Using exact oracle function**: 
```
python3 -m vlab_bench.scripts.run_oracle\
        --func ackley\
        --dims 10\
        --samples 1000\
        --method DOTS
```


- **Using neural network surrogate**: 
```
python3 -m vlab_bench.scripts.run_surrogate\
        --func ackley\
        --dims 10\
        --samples 1000\
        --method DOTS
```

## Running unit tests

## License

The source code is released under the MIT license, as presented in [here](LICENSE).