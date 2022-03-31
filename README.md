# policy-iteration-mlqc
 Policy iteration for multiplicative noise linear quadratic control

## Package dependencies
- numpy
- scipy
- matplotlib


### Conda 

Run the package installation commands

```
conda install numpy scipy matplotlib
```

## Installation

## Cloning the Package
The package contains submodules. Please clone with:
```
git clone https://github.com/TSummersLab/policy-iteration-mlqc.git --recurse
```
Note: When using GitHub desktop, the application initializes all submodules when cloning for the first time.
If a submodule is not initialized after pulling changes, please use the Git Bash tool or terminal and run the command

```
git submodule update --init
```

at the root of the repository.

## Usage

Run `main_single.py` to perform experiments over a single pendulum system with varying amounts of multiplicative noise present. 
- This should take a couple minutes at most to run and should produce plots with various metrics (the same metrics depicted in the paper).

Run `main_multi.py` to perform experiments over multiple problem instances with randomized problem data parameters.
- This may take several minutes to run with the default 1000 trials.
- This should produce a scatter plot of the relative performance of the proposed policy iteration algorithm vs the existing value iteration algorithm.
- The `run_experiments_from_scratch` variable can be toggled to load & analyze previously generated & saved results files. 
