import pandas as pd
import numpy as np
import ast


def find_shifts(l):

    shifts = []
    for i in range(1, len(l)):
        if l[i] == 1 and l[i - 1] == 0:
            shifts.append(i)
    
    return shifts

def find_peaks_custom(signal):
    peaks = []
    n = len(signal)
    
    # Compute the discrete difference (first derivative)
    derivative = np.diff(signal)

    i = 1
    while i < n-1:
        # Peak detected when derivative changes from positive to zero or negative
        if derivative[i-1] > 0 and derivative[i] <= 0:
            # Check for a plateau
            if derivative[i] == 0:
                peaks.append(i)
                while i < n-1 and derivative[i] == 0:
                    i += 1
            else:
                peaks.append(i)
        i += 1

    #if the derivative is always greater than 0, append the last point
    if np.argmax(signal) == n-1:
        peaks.append(n-1)
        return np.array(peaks)
    
    return np.array(peaks)



def read_df_agents(agents_file_path):

    columns_to_read = ["Step", "AgentID", "tipo", "source", "pos"]

    df = pd.read_csv(agents_file_path, usecols = columns_to_read)
    #print("read")
    #cast pos to tuple
    df["pos"] = df["pos"].apply(lambda x: eval(x))
    #print("pos")
    #df["new_position"] = df["new_position"].apply(lambda x: eval(x))
    df["source"] = df["source"].apply(lambda x: eval(x))
    #print("source")

    #reorder columns
    df = df[["Step", "AgentID", "tipo", "source", "pos"]]

    return df


def get_df_flows(df_edges_part):
    df_flow = df_edges_part.groupby(["Step", "source", "pos"]).count()
    df_flow.reset_index(inplace=True)
    df_flow.rename(columns={"AgentID": "flow"}, inplace=True)
    return df_flow


def get_df_edges(df):
    df_edges_A = df[df["tipo"] == "A"][["Step", "AgentID", "source", "pos"]].reset_index(drop=True)
    df_edges_B = df[df["tipo"] == "B"][["Step", "AgentID", "source", "pos"]].reset_index(drop=True)
    df_edges_C = df[df["tipo"] == "C"][["Step", "AgentID", "source", "pos"]].reset_index(drop=True)

    return df_edges_A, df_edges_B, df_edges_C





def get_inouttot_df_flows(num_steps, all_sources, outflows_A, outflows_B, outflows_C, inflows_A, inflows_B, inflows_C, totflows_A, totflows_B, totflows_C):
    outflows_df_A, outflows_df_B, outflows_df_C = pd.DataFrame(index=all_sources), pd.DataFrame(index=all_sources), pd.DataFrame(index=all_sources)
    inflows_df_A, inflows_df_B, inflows_df_C = pd.DataFrame(index=all_sources), pd.DataFrame(index=all_sources), pd.DataFrame(index=all_sources)
    totflows_df_A, totflows_df_B, totflows_df_C = pd.DataFrame(index=all_sources), pd.DataFrame(index=all_sources), pd.DataFrame(index=all_sources)

    for i in range(num_steps):
        outflows_df_A[i] = outflows_A[i]["flow"].values
        outflows_df_B[i] = outflows_B[i]["flow"].values
        outflows_df_C[i] = outflows_C[i]["flow"].values

        inflows_df_A[i] = inflows_A[i]["flow"].values
        inflows_df_B[i] = inflows_B[i]["flow"].values
        inflows_df_C[i] = inflows_C[i]["flow"].values

        totflows_df_A[i] = totflows_A[i]["flow"].values
        totflows_df_B[i] = totflows_B[i]["flow"].values
        totflows_df_C[i] = totflows_C[i]["flow"].values

    outflows_df_A = outflows_df_A.rename(columns=lambda x: x + 1)
    outflows_df_B = outflows_df_B.rename(columns=lambda x: x + 1)
    outflows_df_C = outflows_df_C.rename(columns=lambda x: x + 1)

    inflows_df_A = inflows_df_A.rename(columns=lambda x: x + 1)
    inflows_df_B = inflows_df_B.rename(columns=lambda x: x + 1)
    inflows_df_C = inflows_df_C.rename(columns=lambda x: x + 1)

    totflows_df_A = totflows_df_A.rename(columns=lambda x: x + 1)
    totflows_df_B = totflows_df_B.rename(columns=lambda x: x + 1)
    totflows_df_C = totflows_df_C.rename(columns=lambda x: x + 1)

    outflows_df_A = outflows_df_A.T
    outflows_df_B = outflows_df_B.T
    outflows_df_C = outflows_df_C.T

    inflows_df_A = inflows_df_A.T
    inflows_df_B = inflows_df_B.T
    inflows_df_C = inflows_df_C.T

    totflows_df_A = totflows_df_A.T
    totflows_df_B = totflows_df_B.T
    totflows_df_C = totflows_df_C.T

    return outflows_df_A, outflows_df_B, outflows_df_C, inflows_df_A, inflows_df_B, inflows_df_C, totflows_df_A, totflows_df_B, totflows_df_C


def calculate_count_pivots(df):
    df_count = df.copy()
    df_count.drop(columns=["pos"], inplace=True)
    
    df_count_grouped = df_count.pivot_table(index=["Step", "source"], columns="tipo", aggfunc="size", fill_value=0).reset_index()
    df_count_grouped.rename(columns={"A": "count_A", "B": "count_B", "C": "count_C"}, inplace=True)
    df_count_grouped.columns.name = None

    df_count_grouped_A_pivot = df_count_grouped.pivot(index="Step", columns="source", values="count_A").fillna(0)
    df_count_grouped_B_pivot = df_count_grouped.pivot(index="Step", columns="source", values="count_B").fillna(0)
    df_count_grouped_C_pivot = df_count_grouped.pivot(index="Step", columns="source", values="count_C").fillna(0)

    df_count_grouped_A_pivot.index.name = None
    df_count_grouped_B_pivot.index.name = None
    df_count_grouped_C_pivot.index.name = None

    df_count_grouped_A_pivot.columns.name = None
    df_count_grouped_B_pivot.columns.name = None
    df_count_grouped_C_pivot.columns.name = None

    return df_count_grouped_A_pivot, df_count_grouped_B_pivot, df_count_grouped_C_pivot





def calculate_gent_measure_net_avg_prod(df_net_inflows_A, df_net_inflows_B, df_net_outflows_C, delta, h):
    # Calculate net inflows for A and B combined
    net_inflows_df_AB = df_net_inflows_A + df_net_inflows_B

    # Create an empty DataFrame to store temporal means for net inflows AB
    net_inflows_df_AB_temporal = pd.DataFrame(index=net_inflows_df_AB.index, columns=net_inflows_df_AB.columns)

    # Calculate temporal mean for net inflows AB
    for col in net_inflows_df_AB_temporal.columns:
        for i in range(len(net_inflows_df_AB_temporal)):
            if i >= delta and i >= h:
                mean = net_inflows_df_AB[col].values[i-delta:i].mean()
                if mean < 0:
                    mean = 0
                net_inflows_df_AB_temporal[col].values[i] = mean

    # Create an empty DataFrame to store temporal means for net outflows C
    net_outflows_df_C_temporal = pd.DataFrame(index=df_net_outflows_C.index, columns=df_net_outflows_C.columns)

    # Calculate temporal mean for net outflows C
    for col in net_outflows_df_C_temporal.columns:
        for i in range(len(net_outflows_df_C_temporal)):
            if i >= delta and i >= h:
                mean = df_net_outflows_C[col].values[i-delta:i].mean()
                if mean < 0:
                    mean = 0
                net_outflows_df_C_temporal[col].values[i] = mean


    # Calculate the product of temporal net inflows AB and net outflows C
    df_prod_temporal_net = (net_inflows_df_AB_temporal * net_outflows_df_C_temporal) 
    #take the sqrt of the product
    df_prod_temporal_net = df_prod_temporal_net.applymap(lambda x: np.sqrt(x)  if np.isfinite(x) else x)


    return df_prod_temporal_net





def calculate_gent_dummy_measure(df_change_pivot, prop_AB, delta,h):
    # Create an empty DataFrame to store temporal differences for change pivot
    df_change_pivot_temporal = pd.DataFrame(index = df_change_pivot.index, columns = df_change_pivot.columns)

    for col in df_change_pivot_temporal.columns:
        for i in range(0, len(df_change_pivot_temporal)):
            if i >= delta and i >= h:
                values_delta =  df_change_pivot[col].values[i-delta:i]
                
                mean = values_delta.mean()
                if mean > prop_AB:
                    df_change_pivot_temporal[col].values[i] = 1
                else:
                    df_change_pivot_temporal[col].values[i] = 0

    return df_change_pivot_temporal









def str_to_np_array(s):
    # Remove unnecessary characters and split the string into lists of floats
    list_of_lists = [[float(num) for num in sublist.split()] for sublist in s[2:-2].replace('\n', '').split('] [')]
    # Convert list of lists to NumPy array
    return np.array(list_of_lists)

def str_to_dict(input_str):

    result_dict = ast.literal_eval(input_str)
    return result_dict



def fill_with_missing_pairs(aggr_flow, all_sources, how = "out"):

    if how == "out":
        node = "source"
    elif how == "in":
        node = "pos"
    
    existing_nodes= aggr_flow[node].unique()
    missing_nodes = list(set(all_sources) - set(existing_nodes))


    # add the missing sources to the outflow dataframe
    for nd in missing_nodes:
        aggr_flow = pd.concat([aggr_flow, pd.DataFrame([[nd, 0]], columns=[node, 'flow'])])

    # sort the dataframe by source
    aggr_flow = aggr_flow.sort_values(by=node).reset_index(drop=True)

    return aggr_flow


def get_out_in_tot_flows(df_flow_tipo, all_sources, num_steps):

    outflows_tipo = []
    inflows_tipo = []
    totflows_tipo = []

    #for step in tqdm(range(1,num_steps+1)):
    for step in range(1,num_steps+1):
            
        df_flow_tipo_step = df_flow_tipo[df_flow_tipo["Step"] == step]

        outflow_tipo_step = df_flow_tipo_step.groupby("source")["flow"].sum().reset_index()
        outflow_tipo_step = fill_with_missing_pairs(outflow_tipo_step, all_sources, how = "out")
        outflows_tipo.append(outflow_tipo_step)

        inflow_tipo_step = df_flow_tipo_step.groupby("pos")["flow"].sum().reset_index()
        inflow_tipo_step = fill_with_missing_pairs(inflow_tipo_step, all_sources, how = "in")
        inflows_tipo.append(inflow_tipo_step)

        totflow_tipo_step = inflow_tipo_step.copy()
        totflow_tipo_step["flow"] = inflow_tipo_step["flow"] + outflow_tipo_step["flow"]
        totflows_tipo.append(totflow_tipo_step)

    return outflows_tipo, inflows_tipo, totflows_tipo




def get_df_flow(df_edges_tipo):
    
    df_flow_tipo = df_edges_tipo.groupby(["Step", "source", "pos"]).count()
    df_flow_tipo.reset_index(inplace = True)
    df_flow_tipo.rename(columns = {"AgentID": "flow"}, inplace = True)

    return df_flow_tipo



def get_df_out_in_tot_flows(outflows, inflows, totflows, all_sources, num_steps):
    """Initialize, populate, rename, and transpose flow DataFrames for a specific tipo."""
    
    # Initialize empty DataFrames for outflows, inflows, and total flows
    outflows_df = pd.DataFrame(index=all_sources)
    inflows_df = pd.DataFrame(index=all_sources)
    totflows_df = pd.DataFrame(index=all_sources)
    
    # Populate the DataFrames with flow data
    for i in range(num_steps):
        outflows_df[i] = outflows[i]["flow"].values
        inflows_df[i] = inflows[i]["flow"].values
        totflows_df[i] = totflows[i]["flow"].values

    # Rename the columns
    outflows_df = outflows_df.rename(columns=lambda x: x + 1)
    inflows_df = inflows_df.rename(columns=lambda x: x + 1)
    totflows_df = totflows_df.rename(columns=lambda x: x + 1)

    # Transpose the DataFrames
    outflows_df = outflows_df.T
    inflows_df = inflows_df.T
    totflows_df = totflows_df.T

    return outflows_df, inflows_df, totflows_df


