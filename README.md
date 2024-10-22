# Gentrification

# Dynamic models of gentrification
## Table of contents
1. [Citing](#citing)
2. [Packages](#packages)
3. [Abstract](#abstract)
4. [Analysis](#analysis)



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


# Analysis

# Analysis

### Single Run Analysis
The `run_single.ipynb` notebook provides:
- Interactive model experimentation 
- Detailed visualization of model dynamics
- Exploratory analysis capabilities

### Batch Processing
Run multiple simulations with different parameters using `run_batch.py`:

`python run_batch.py <strategy> <deployment>`

#### Parameters:
- `strategy`: 
  - `improve`: Main model (primary implementation)
  - `random`: Full Random model 
  - `randomdest`: Random Destination model
- `deployment`:
  - `centre_segr`: Standard central segregation
  - `centre_segr-big`: Extended central segregation


#### Output Structure
Results location:

out/batch_results/{strategy}/{deployment}/{n_agents}/exps/

Generated for each run:
- Model DataFrame: Global model evolution metrics
- Agent DataFrame: Time-series of agent locations and properties

### Post-Processing
Calculate segregation metrics from batch results:

`python batch_calculate_Gs.py -m <mode> -n <num_agents>`

#### Parameters:
- `-m`: Analysis mode (`improve`, `random`, `randomdest`)
- `-n`: Agent count to analyze

#### Output
Saved to:

out/batch_results/{mode}/{deployment}/{n_agents}/intermediate/

Producing:
1. Network-based segregation metrics
2. Count-based segregation metrics


**Warning** Both `run_batch.py` and `batch_calculate_Gs.py` are built with an highly-parallelised code, given the high computational load of the task.




### Results Analysis

#### Main Results
Use `batch_analysis.ipynb` to reproduce Figures 3,5,6

#### Flow Visualization
Use `flows_viz.ipynb` to generate Figure 4a, b

#### Early Warning Analysis
Use `who_first.ipynb` to reproduce: Figure 4c and Supplementary Notes 1-3: 

## Project Structure
```
├── run_single.ipynb      # Single run analysis
├── batch_analysis.ipynb  # First results visualization
├── flows_viz.ipynb      # Flow pattern analysis
|── who_first.ipynb      # Early warning analysis
|── run_batch.py         # Batch runner
└── batch_calculate_Gs.py # Segregation metrics calculator
out/
  └── batch_results/       # Results storage
      ├── improve/        
      ├── random/
      └── randomdest/
```