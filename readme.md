# Installation
Since this project relies on Pytorch (https://pytorch.org), future compatibility is not guaranteed.
The environment on which it is based is found in the file conda_env_specs.txt

To clone it into a conda env :

* Make sure you are using a 64-bits linux (if not, could work but packages compatibility not guaranteed)
* If you do not already have any Anaconda flavor installed, download and install Miniconda 3.7 64 bits (https://conda.io/miniconda.html). I recommend making it the default python when asked for it if you don't want to have
to think about which python you're calling.
* Download or clone the project, *e.g.* as 'gmmc', and move to that directory from command line.
* Run :
```
conda install conda_env_specs.txt
```
if you want to clone this environment as your default, or :
```
conda create --name myenv conda_env_specs.txt
```
replacing "myenv" with any appropriate name if you want this as a new env (if you chose the second way, don't forget to activate the environment before running the code)

* Still in the 'gmmc' directory, run :
```
python setup.py
```
This will create the directories that were not included in the git but are needed to run the program.

# Reproducing experiments

Simply run from command line :
```
python full_MC.py
```

Relevant parameters are given at the beginning of that script (be careful about n_threads, 12 is a lot for non-professional CPUs and could very likely cause a crash either of Python or the full system...)
