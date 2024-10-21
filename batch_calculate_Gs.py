from operator import index
import os
import sys
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from utils_gent_measure import *
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='batch_measure_run.log', filemode='w')




num_agents = 2**12  # Set the number of agents (default: 2^12)
if '-n' in sys.argv:
    num_agents = int(sys.argv[sys.argv.index('-n') + 1])

else:
    print("Number of agents not specified. Defaulting to 2**12")

if '-m' in sys.argv:
    mode = sys.argv[sys.argv.index('-m') + 1]
    if mode not in ["improve", "random", "randomdest"]:
        raise ValueError("Invalid mode. Please choose from 'improve', 'random', 'randomdest'")
    
else:
    mode = "improve"
    print("Mode not specified. Defaulting to 'improve'")


starting_deployment = "centre_segr"  # Set the starting deployment (default: "centre_segr")

directory = f"out/batch_results/{mode}/{starting_deployment}-big/{num_agents}agents/exps"

p_g_list =[round(i * 0.05, 2) for i in range(4)]  # List of p_g values in the folder
p_g_list.insert(1, 0.01)

if mode == "random":
    p_g_list = [0.01]

print(p_g_list)
h_values = [20]
rep_values = list(range(0, 150))
delta_list = [10,15,20]  # List of delta values



def process_rep(p_g, h, rep, delta):
    p_g_str = f"pg_{p_g}_"
    h_str = f"h_{h}_"
    rep_str = f"rep_{rep}_"
    delta_str = f"delta_{delta}_"

    for file_name in os.listdir(directory):
        if  p_g_str in file_name and h_str in file_name and rep_str in file_name:
            if file_name.endswith("_results_agents.csv"):

                file_name_model = file_name.replace("_results_agents.csv", "_results_model.csv")

                agents_file_path = os.path.join(directory, file_name)

                df_agents_raw = pd.read_csv(agents_file_path)
                vc = df_agents_raw[df_agents_raw["Step"]==1]["tipo"].value_counts()
                prop_A = vc["A"] / num_agents
                prop_B = vc["B"] / num_agents
                prop_C = vc["C"] / num_agents

                prop_AB = prop_A + prop_B


                df = read_df_agents(agents_file_path)



                ###### counts
                ######
                ######
                ######
                df_count_grouped_A_pivot, df_count_grouped_B_pivot, df_count_grouped_C_pivot = calculate_count_pivots(df)
                #####



                #####flows
                #####
                #####
                #####
                df_edges = df[df["source"] != df["pos"]].reset_index(drop=True)
                df_edges_A, df_edges_B, df_edges_C = get_df_edges(df_edges)

                df_flow_A = get_df_flows(df_edges_A)
                df_flow_B = get_df_flows(df_edges_B)
                df_flow_C = get_df_flows(df_edges_C)

                mass = range(df["pos"].max()[0] + 1)
                num_steps = max(df["Step"])
                


                all_sources = [(o, d) for o in mass for d in mass]
                outflows_A, inflows_A, totflows_A = get_out_in_tot_flows(df_flow_A, all_sources, num_steps)
                outflows_B, inflows_B, totflows_B = get_out_in_tot_flows(df_flow_B, all_sources, num_steps)
                outflows_C, inflows_C, totflows_C = get_out_in_tot_flows(df_flow_C, all_sources, num_steps)

                outflows_df_A, outflows_df_B, outflows_df_C, inflows_df_A, inflows_df_B, inflows_df_C, totflows_df_A, totflows_df_B, totflows_df_C = get_inouttot_df_flows(
                    num_steps, all_sources, outflows_A, outflows_B, outflows_C, inflows_A, inflows_B, inflows_C, totflows_A, totflows_B, totflows_C)

                df_net_outflows_A = outflows_df_A - inflows_df_A
                df_net_outflows_B = outflows_df_B - inflows_df_B
                df_net_outflows_C = outflows_df_C - inflows_df_C

                df_net_inflows_A = inflows_df_A - outflows_df_A
                df_net_inflows_B = inflows_df_B - outflows_df_B
                df_net_inflows_C = inflows_df_C - outflows_df_C


                
                df_norm_net_outflows_A = df_net_outflows_A / (outflows_df_A + outflows_df_B + outflows_df_C)
                df_norm_net_outflows_B = df_net_outflows_B / (outflows_df_A + outflows_df_B + outflows_df_C)
                df_norm_net_outflows_C = df_net_outflows_C / (outflows_df_A + outflows_df_B + outflows_df_C)

                df_norm_net_inflows_A = df_net_inflows_A / (inflows_df_A + inflows_df_B + inflows_df_C)
                df_norm_net_inflows_B = df_net_inflows_B / (inflows_df_A + inflows_df_B + inflows_df_C)
                df_norm_net_inflows_AB = df_norm_net_inflows_A + df_norm_net_inflows_B
                df_norm_net_inflows_C = df_net_inflows_C / (inflows_df_A + inflows_df_B + inflows_df_C)
                


                # in df_norm_net_outflows_C and df_norm_net_inflows_AB  substituthe the negative values with 0
                df_norm_net_outflows_C[df_norm_net_outflows_C < 0] = 0
                df_norm_net_inflows_AB[df_norm_net_inflows_AB < 0] = 0



                # Save the DataFrames with the coherent naming convention
                #remove the exps from the directory path
                intermediate_dir = f"out/batch_results/{mode}/{starting_deployment}-big/{num_agents}agents/intermediate"
                os.makedirs(intermediate_dir, exist_ok=True)



                ###Dummy measure
                df_change_pivot = (df_count_grouped_A_pivot + df_count_grouped_B_pivot) / ( df_count_grouped_A_pivot +df_count_grouped_B_pivot + df_count_grouped_C_pivot)
                df_chi = df_change_pivot.copy()
                #Apply the rolling operation starting from the h-th row
                df_chi.iloc[h:] = df_chi.iloc[h:].rolling(window=delta).mean()

                df_chi_hat = calculate_gent_dummy_measure(df_change_pivot, prop_AB, delta,h)


                # first measure ATTENTION IS NORMALIZED
                #in the normalized data substitute nans with 0s
                df_norm_net_inflows_A.fillna(0, inplace=True)
                df_norm_net_inflows_B.fillna(0, inplace=True)
                df_norm_net_outflows_C.fillna(0, inplace=True)



                df_chi_hat.to_csv(os.path.join(intermediate_dir, f"{p_g_str}{h_str}{rep_str}{delta_str}results_chi_hat.csv"))
                df_chi.to_csv(os.path.join(intermediate_dir, f"{p_g_str}{h_str}{rep_str}{delta_str}results_chi.csv"))



                df_net_avg_prod = calculate_gent_measure_net_avg_prod(df_norm_net_inflows_A, df_norm_net_inflows_B, df_norm_net_outflows_C, delta, h)


                df_net_avg_prod.to_csv(os.path.join(intermediate_dir, f"{p_g_str}{h_str}{rep_str}{delta_str}results_net_avg_prod.csv"))


                logging.info(f"Processed: n_agents: {num_agents}, p_g value: {p_g}, h value: {h}, rep value: {rep}, delta value: {delta} done")

                
    return p_g, h, rep, delta

# Generate the combinations of p_g, h, rep, and delta
tasks = [(p_g, h, rep, delta) for p_g in p_g_list[::-1] for h in h_values for rep in rep_values for delta in delta_list]
print(len(tasks))

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_rep, p_g, h, rep, delta) for p_g, h, rep, delta in tasks]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
        p_g, h, rep, delta = future.result()
        # Process the result if needed

