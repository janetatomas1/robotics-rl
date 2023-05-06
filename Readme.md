
robotics-rl
==========

Implementation of reinforcement learning environment for robotic arms supported by the CoppeliaSim simulator.

Tested on:
- Operation system - Ubuntu 20.04/22.04\
- [CoppeliaSim 4.4.0](https://www.coppeliarobotics.com/downloads)
- Python 3.10

Used libraries:
- [gym](https://github.com/openai/gym) - framework for creating and managing Rl environments\
- [stables-baselines3](https://github.com/DLR-RM/stable-baselines3) - library containing latest reinforcement learning algorithms
- [Pyrep](https://github.com/stepjam/PyRep) - library for controlling the CoppeliaSim simulator


Install on desktop
================

- download CoppeliaSim from the link above as a compressed file
- extract it on your computer
- add following lines to your bash profile:
```
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```
- create a Python virtual environment via virtualenv command
- install all requirements from the file `requirements/dev.txt`

For the last step, you may need to downgrade to `setuptools==66`. The issue is caused by using an old version of the gym package.

To see all possible options, run `python -m roboticsrl.main --help`


Deploy
===============
To install on a server, create Python virtual environment and install requirements from the file `requirements/production.txt`.
After this, simply run `./scripts/start-docker.sh`.

All results of training and evaluation will be saved in `$HOME/results/$CONTAINER_NAME` directory.
Trained models with the best performance are saved in the `models` directory, and the `positions` directory
contains testing data for evaluating effectiveness.


Demo
=====
To run demo of the trained models, simply start the `roboticsrl/demo.py` script.
