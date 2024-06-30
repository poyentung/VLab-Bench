from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "0.0.1"
    
setup(
    name='vlab_bench',
    version=__version__,
    description="VLab-Bench: Benchmarks for self-driving virtual laboratories",
    license='MIT',
    url="https://github.com/poyentung/VLab-Bench",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "notebook",
        "tensorflow",
        "keras",
        "numpy",
        "hydra-core         >= 1.3",
        "scipy              == 1.10.1",
        "pandas             == 1.4.4", 
        "matplotlib         == 3.6.3",
        "matplotlib-inline  == 0.1.6",
        "scikit-learn       == 1.2.2",
        "scikit-image       == 0.19.3",
        "cma                == 3.3.0",
        "nevergrad                  ",
        "tqdm               == 4.59.0",
        "seaborn            == 0.12.2",
        "openpyxl           == 3.1.2",
    ],
    package_data={
        "": ["LICENSE", "README.md"]
    },
)