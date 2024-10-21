import pandas as pd
import numpy as np
from ABM_MeanField_Cells.model import GentModel
from concurrent.futures import ProcessPoolExecutor
import os
import itertools
import random
import warnings
import logging
import sys
from tqdm.auto import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None

SAVEDATA = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='batch_model_run.log', filemode='w')

#take the parameter mode from the terminal. If not specified, the default is "improve"

if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    mode = "improve"



# Define fixed and variable parameters
fixed_parameters = {
    'num_steps': 300,
    'width': 9,
    'height': 9,
    'mode': mode,
    'starting_deployment': "centre_segr",
    'empty_border': 0,
    'seed': 67,
    'tqdm_on': False,
    'h': 20,
    'termination_condition': True,
}

variable_parameters = {
    'p_g': [0.01] + [round(i * 0.05, 2) for i in range(4)] if mode != "random" else [0.01],
    'num_agents': [2**x for x in range(7, 13)],
}

# Number of repetitions per parameter set
repetitions = 150

def generate_parameter_combinations(variable_params, repetitions):
    """Generate all combinations of variable parameters with repetitions."""
    keys, values = zip(*variable_params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    parameter_combinations = [(rep, comb) for comb in combinations for rep in range(repetitions)]
    return parameter_combinations

def run_model_iteration(args):
    repetition_index, variable_params = args
    params = {**fixed_parameters, **variable_params}
    


    # Define the directory and file paths
    directory = f"out/batch_results/{params['mode']}/{params['starting_deployment']}-big/{params['num_agents']}agents/exps"
    # Create a directory for the results if it does not exist
    os.makedirs(directory, exist_ok=True)

    model_file_path = f"{directory}/pg_{params['p_g']}_h_{params['h']}_rep_{repetition_index}_results_model.csv"
    agents_file_path = f"{directory}/pg_{params['p_g']}_h_{params['h']}_rep_{repetition_index}_results_agents.csv"

    # Check if the files already exist
    if os.path.exists(model_file_path) and os.path.exists(agents_file_path):
        logging.info(f"Files for params {variable_params} repetition {repetition_index} already exist. Skipping.")
        return




    # Initialize the model with combined parameters
    model = GentModel(
        num_agents=params['num_agents'],
        width=params['width'],
        height=params['height'],
        mode=params['mode'],
        starting_deployment=params['starting_deployment'],
        p_g=params['p_g'],
        h=params['h'],
        empty_border=params['empty_border'],
        seed=params['seed'],
        tqdm_on=params['tqdm_on'],
        termination_condition=params['termination_condition']
    )
    



    # Run the model
    model.run_model(params['num_steps'])
    
    # Get the df of the datacollector
    df = model.datacollector.get_model_vars_dataframe()
    df_agents = model.datacollector.get_agent_vars_dataframe()

    # Save the data in the folder batch_results. If the folder does not exist, create it
    os.makedirs(directory, exist_ok=True)

    # Save dataframes to CSV files
    df.to_csv(model_file_path)
    df_agents.to_csv(agents_file_path)
    
    logging.info(f"Model with params {variable_params} repetition {repetition_index} saved")

def run_parallel(parameter_combinations):
    num_iterations = len(parameter_combinations)

    with ProcessPoolExecutor() as executor:
        # Use tqdm to display the progress bar
        list(tqdm(executor.map(run_model_iteration, parameter_combinations), total=num_iterations))

# Generate parameter combinations
parameter_combinations = generate_parameter_combinations(variable_parameters, repetitions)
print(variable_parameters['p_g'])
#print the number of parameter combinations
print(len(parameter_combinations))

# Shuffle the parameter_combinations list to randomize the order of parameter sets (just for balancing the load)
random.shuffle(parameter_combinations)

# Run the main function
run_parallel(parameter_combinations)
