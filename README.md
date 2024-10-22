# Gentrification

# Dynamic models of gentrification
## Table of contents
1. [Citing](#citing)
2. [Packages](#packages)
3. [Abstract](#abstract)
5. [Structure of the repository](#structure-of-the-repository)
6. [Analysis](#analysis)



# Citing
In this repository you can find the code for running our Gentrification model and to replicate the analysis conducted in our paper.
If you use the code in this repository, please cite our paper:

*Mauro, G., Pedreschi, N., Pappalardo, L, Lambiotte,R. (2024). Dynamic models of gentrification. arXiv preprint arXiv:XXXX (2024).

```
@article{mauro2024dynamic,
  title={Dynamic models of gentrification},
  author={Mauro, Giovanni and Pedreschi, Nicola and Pappalardo, Luca and Lambiotte, Renaud.},
  journal={XXX},
  volume={XX},
  number={XX},
  pages={XX},
  year={2024},
  publisher={XXX}
}
```

# Packages
For running notebooks and scripts of this project you must install the following Python packages:
```
  mesa
  networkx
  seaborn
  numpy
  jupyter
```
These packages will automatically install the other required ones (e.g ```matplotilib``` etc.).

# Abstract

The phenomenon of gentrification of an urban area is characterized by the displacement of lower-income residents due to rising living costs and an influx of wealthier individuals. This study presents an agent-based model simulating urban gentrification through the relocation of three income groups driven by living costs. The model incorporates economic and sociological theories to generate realistic neighborhood transition patterns. We introduce a temporal network-based measure to track the outflow of low-income residents and the inflow of middle- and high-income residents over time. Numerical experiments reveal that high-income residents trigger gentrification. Our network-based measure consistently detects gentrification patterns earlier than traditional count-based methods, potentially serving as an early detection tool in real-world scenarios. The analysis also highlights how city density promotes gentrification. This framework offers valuable insights for understanding gentrification dynamics and informing urban planning and policy decisions.


# Structure of the repository

In the **main** level of the repo you can find:
- ```run_single.ipynb```and ```run_batch.py```
    - These codes execute, respectively, one run of the model and a batch execution of more models (more repetitions, several parameter combintions).
        - ```run_single.ipynb```
           Notebook for taking familiarity with the model and showing some interesting plots (not in the paper)
        - ```run_batch.py```
        -  Script for running several parameter combination and repetitions of the model. Take one string as parameter, the strategy of agents. Can be ```improve```, ```random``` or ```randomdest```. The first one is the main model, while the other two are, respectively, the Full Random model and the Random Destination model (see Supplementary Notes for more details).

