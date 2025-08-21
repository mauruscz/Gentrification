import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
import ast





def calculate_neighborhood_capacity(num_agents, num_neighborhoods, occupancy_rate=0.75):
    total_capacity = num_agents / occupancy_rate
    capacity_per_neighborhood = total_capacity / num_neighborhoods
    return round(capacity_per_neighborhood)


def calculate_cells2wealths(model):
    cell_list = [cell[1] for cell in model.grid.coord_iter()]
    cells2wealths = {cell: [a.wealth for a in model.grid.get_cell_list_contents(cell)] for cell in cell_list}
    return cells2wealths


def calculate_cells2types(model):
    cell_list = [cell[1] for cell in model.grid.coord_iter()]
    cells2types = {cell: [a.tipo for a in model.grid.get_cell_list_contents(cell)] for cell in cell_list}
    return cells2types



def calculate_arriving_percentile(model, agent, cells_with_space): #calculate the percentile of the agent wealth in the cells with space

    cell_list = cells_with_space.copy()
    #drop the cell where the agent is, if it is in the list
    if agent.pos in cell_list:
        cell_list.remove(agent.pos)

    cells2wealths = {cell: [a.wealth for a in model.grid.get_cell_list_contents(cell)] for cell in cell_list}


    cells2percentile = {
        cell: min(100.0, round(percentileofscore(cells2wealths[cell], agent.wealth, kind='weak'), 12))
        for cell in cell_list
    }

    #if there's a value >100 print it
    for cell in cells2percentile:
        if cells2percentile[cell] > 100:
            cells2percentile[cell] = 100
        if cells2percentile[cell] < 0:
            cells2percentile[cell] = 0

    return cells2percentile


def calculate_arriving_percentile_2(model, agent): #same of before but using the cells2wealths dictionary

    cell_list = [cell[1] for cell in model.grid.coord_iter()]

    #drop the cell where the agent is
    cell_list.remove(agent.pos)

    cells2percentile = {cell: percentileofscore(model.cells2wealths[cell], agent.wealth, kind='weak') for cell in cell_list}


    return cells2percentile


def eligibles_cells_with_space(model, eligibles):
    
    eligibles_with_space = []
    
    for cell in eligibles:
        
        if model.grid.is_cell_empty(cell):
            eligibles_with_space.append(cell)
        elif len( list(model.grid.iter_cell_list_contents(cell)) )    <           model.capacity[cell[0]][cell[1]]:
            eligibles_with_space.append(cell)

    return eligibles_with_space




def calculate_percentile_richness_matrix(model, percentile):
    percentile_richness_matrix = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        x = cell[1][0]
        y = cell[1][1]
        percentile_richness_matrix[x][y] = calculate_percentile_richness(model, x, y, percentile)
    return percentile_richness_matrix

def calculate_percentile_richness(model, x, y, percentile):
    #calculate the percentile wealth of the cell, using the function iter_cell_list_contents. if the cell is empty, return 0

    if model.grid.is_cell_empty([x,y]):
        return 0
    else:
        percentile_curr_node_wealth = np.percentile([agent.wealth for agent in model.grid.iter_cell_list_contents([x,y])], percentile)
        return percentile_curr_node_wealth



def calculate_median_richness_matrix(model):
    median_richness_matrix = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        x = cell[1][0]
        y = cell[1][1]
        median_richness_matrix[x][y] = calculate_median_richness(model, x, y)
    return median_richness_matrix

def calculate_median_richness(model, x, y):
    #calculate the median wealth of the cell, using the function iter_cell_list_contents. if the cell is empty, return 0

    if model.grid.is_cell_empty([x,y]):
        return 0
    else:
        median_curr_node_wealth = np.median([agent.wealth for agent in model.grid.iter_cell_list_contents([x,y])])
        return median_curr_node_wealth


def calculate_std_dev_richness_matrix(model):
    std_dev_richness_matrix = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        x = cell[1][0]
        y = cell[1][1]
        std_dev_richness_matrix[x][y] = calculate_std_dev_richness(model, x, y)
    return std_dev_richness_matrix

def calculate_std_dev_richness(model, x, y):
    #calculate the std dev of the wealth of the cell, using the function iter_cell_list_contents. if the cell is empty, return 0

    if model.grid.is_cell_empty([x,y]):
        return 0
    else:
        std_dev_curr_node_wealth = np.std([agent.wealth for agent in model.grid.iter_cell_list_contents([x,y])])
        return std_dev_curr_node_wealth


def calculate_mean_richness_matrix(model):
    mean_richness_matrix = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        x = cell[1][0]
        y = cell[1][1]
        mean_richness_matrix[x][y] = calculate_mean_richness(model, x, y)
    return mean_richness_matrix

def calculate_mean_richness(model, x, y):
    #calculate the average wealth of the cell, using the function iter_cell_list_contents. if the cell is empty, return 0

    if model.grid.is_cell_empty([x,y]):
        return 0
    else:
        mean_curr_node_wealth = np.mean([agent.wealth for agent in model.grid.iter_cell_list_contents([x,y])])
        return mean_curr_node_wealth
    

def calculate_total_richness_matrix(model):
    total_richness_matrix = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        x = cell[1][0]
        y = cell[1][1]
        total_richness_matrix[x][y] = calculate_total_richness(model, x, y)
    return total_richness_matrix

def calculate_total_richness(model, x, y):
    #calculate the total wealth of the cell, using the function iter_cell_list_contents. if the cell is empty, return 0

    if model.grid.is_cell_empty([x,y]):
        return 0
    else:
        total_curr_node_wealth = np.sum([agent.wealth for agent in model.grid.iter_cell_list_contents([x,y])])
        return total_curr_node_wealth
    


#weighted cells is a dictionary with the key as cell and the value as the weighted probability of that cell
#the function iterates over the cells, picking one at random with a probability equal to the weighted probability
#of that cell. if the cell is full, it removes it from the dictionary and picks another one
#it returns the cell picked, (-1,-1) if no cell was picked
def find_a_spot_in_weighted_cells(model, weighted_cells):
        
        
        #if the dictionary is empty, return -1
        if len(weighted_cells) == 0:
            return (-1,-1)
        
        #pick a cell at random with a probability equal to the weighted probability of that cell
        cell = model.random.choices(list(weighted_cells.keys()), weights = list(weighted_cells.values()))[0]
        
        #if the cell is full, remove it from the dictionary and pick another one
        while len( list(model.grid.iter_cell_list_contents(cell)) ) >= model.capacity[cell[0]][cell[1]]:
            del weighted_cells[cell]
            
            #if the dictionary is empty, return -1
            if len(weighted_cells) == 0:
                return (-1,-1)
            
            #pick a cell at random with a probability equal to the weighted probability of that cell
            #cell = model.random.choices(list(weighted_cells.keys()), weights = list(weighted_cells.values()))[0]
            cell = model.random.choices(list(weighted_cells.keys()), weights = list(weighted_cells.values()))[0]

        return cell







def average_growth_rate(arr):
    total_growth = 0
    for i in range(1, len(arr)):
        total_growth += arr[i] - arr[i - 1]
    return total_growth / (len(arr) - 1)












#pick a random row with a probability equal to the percent column
def pick_random_row(df, model, percent_cumul_limit_low = 0, percent_cumul_limit_high = 100):

    df1 = df[(df["percent_cumul"] >= percent_cumul_limit_low) & (df["percent_cumul"] <= percent_cumul_limit_high)]

    #df1["percent"] now not sums to 100. make it so in df1
    total_percent = df1["percent"].sum()
    df1.loc[:, "percent"] = df1["percent"] / total_percent

    #print(df1)
    return model.random.choices(df1.index, weights = df1["percent"])[0]

#pick a random amount between the lower and upper bound of a row picked with pick_random_row
def pick_random_amount(df, row, model):
    return model.random.uniform(df["bound_low"][row], df["bound_high"][row])







def fill_with_missing_pairs(aggr_flow, all_sources, how = "out"):

    if how == "out":
        node = "source"
    elif how == "in":
        node = "pos"
    
    existing_nodes= aggr_flow[how].unique()
    missing_nodes = list(set(all_sources) - set(existing_nodes))

    # add the missing sources to the outflow dataframe
    for nd in missing_nodes:
        aggr_flow = pd.concat([aggr_flow, pd.DataFrame([[nd, 0]], columns=[node, 'flow'])])

    # sort the dataframe by source
    aggr_flow = aggr_flow.sort_values(by=node)




def place_agent_2(model, tipo):


    def check_capacity_and_choose(eligible_cells):
        for _ in range(len(eligible_cells)):  # Limit the number of tries to the number of eligible cells
            (x, y) = model.random.choice(eligible_cells)
            if len(model.grid.get_cell_list_contents([(x, y)])) < model.capacity[x][y]:
                return (x, y)
        return None  # Return None if no eligible cell is found

    def place_agent_with_fallback(primary_cells, fallback_cells):
        coord = check_capacity_and_choose(primary_cells)
        if coord:
            return coord
        coord = check_capacity_and_choose(fallback_cells)
        if coord:
            return coord
        raise Exception(f"Unable to place agent of type {tipo} due to capacity constraints")
    



    all_cells = [cell[1] for cell in model.grid.coord_iter()]
    if model.starting_deployment == "random":
        coord = place_agent_with_fallback(all_cells, all_cells)
        if coord:
            return coord
        raise Exception("Unable to place agent due to capacity constraints")

    elif model.starting_deployment == "centre_segr":
        
        centre_coord = (model.grid.width // 2, model.grid.height // 2)
        if model.grid.width % 2 == 0:
            centre_coord = (centre_coord[0] - 1, centre_coord[1])
        if model.grid.height % 2 == 0:
            centre_coord = (centre_coord[0], centre_coord[1] - 1)

        nearby_center_cells = [(centre_coord[0] + dx, centre_coord[1] + dy)
                               for dx in range(-1, 2) for dy in range(-1, 2)
                               if 0 <= centre_coord[0] + dx < model.grid.width
                               and 0 <= centre_coord[1] + dy < model.grid.height]

        eligibles_B = [cell[1] for cell in model.grid.coord_iter()
                       if 0 < cell[1][0] < model.grid.width - 1
                       and 0 < cell[1][1] < model.grid.height - 1
                       and cell[1] != centre_coord]
        eligibles_C = [cell[1] for cell in model.grid.coord_iter()
                       if cell[1][0] < 1 or cell[1][0] >= model.grid.width - 1
                       or cell[1][1] < 1 or cell[1][1] >= model.grid.height - 1]

        if tipo == 'A':
            if len(model.grid.get_cell_list_contents([centre_coord])) < model.capacity[centre_coord[0]][centre_coord[1]]:
                return centre_coord
            else:
                return place_agent_with_fallback(nearby_center_cells, all_cells)
        elif tipo == 'B':
            return place_agent_with_fallback(eligibles_B, all_cells)
        elif tipo == 'C':
            return place_agent_with_fallback(eligibles_C, all_cells)

    raise Exception("Unknown deployment type")















def place_agent(model, tipo,  num_cells_A, num_cells_B):

    if model.empty_border == 0:
        cells_list_coord = [cell[1] for cell in model.grid.coord_iter()]

    else: # the border cells are not eligible
        cells_list_coord = [cell[1] for cell in model.grid.coord_iter()
                            if cell[1][0] >= model.empty_border and cell[1][0] < model.grid.width - model.empty_border
                            and cell[1][1] >= model.empty_border and cell[1][1] < model.grid.height - model.empty_border]
        


    if model.starting_deployment == "corner_segr":


        if tipo == 'A':
            (x,y) = model.random.choice(cells_list_coord[:num_cells_A])

        elif tipo == 'B':
            (x,y) = model.random.choice(cells_list_coord[num_cells_A:num_cells_A + num_cells_B])


        elif tipo == 'C':
            (x,y) = model.random.choice(cells_list_coord[num_cells_A + num_cells_B:])



    elif model.starting_deployment == "random":
        x = model.random.randrange(model.grid.width)
        y = model.random.randrange(model.grid.height)


    elif model.starting_deployment == "centre_segr": 
        centre_coord = (model.grid.width//2, model.grid.height//2)

        eligibles_B = [cell[1] for cell in model.grid.coord_iter() 
                        if 0 < cell[1][0] < model.grid.width - 1
                        and 0 < cell[1][1] < model.grid.height - 1
                        and cell[1] != centre_coord]
        
        eligibles_C = [cell[1] for cell in model.grid.coord_iter()
                        if cell[1][0] < 1 or cell[1][0] >= model.grid.width - 1
                        or cell[1][1] < 1 or cell[1][1] >= model.grid.height - 1]


        if tipo == 'A':
            (x,y) = centre_coord
            
        elif tipo == 'B':
            # Pick a random cell from the eligibles
            (x, y) = model.random.choice(eligibles_B)

        elif tipo == 'C':
            # Pick a random cell from the eligibles
            (x, y) = model.random.choice(eligibles_C)


        
    return (x,y)





def get_median_richness_df(df_exec_model, all_sources):

    df_cell_median = df_exec_model[["median_richness_matrix"]]
    #transform the string into a dictionary
    df_cell_median["median_richness_matrix"] = df_cell_median["median_richness_matrix"].apply(lambda x: str_to_np_array(x))
    #transform the dictionary into a list
    df_cell_median["median_richness_matrix"] = df_cell_median["median_richness_matrix"].apply(lambda x: list(x.flatten()))
    df_cell_median = df_cell_median["median_richness_matrix"].apply(pd.Series)
    #return to a column names of pairs
    df_cell_median.columns = all_sources
    df_cell_median.index = df_cell_median.index + 1

    return df_cell_median


def get_gini_richness_df(df_exec_model, all_sources):

    df_cell_gini = df_exec_model[["Gini"]]
    #transform the string into a dictionary, is already a dictionary
    df_cell_gini["Gini"] = df_cell_gini["Gini"].apply(lambda x: str_to_dict(x))
    df_cell_gini["Gini"] = df_cell_gini["Gini"].apply(lambda x: dict(sorted(x.items())))

    #transform the dictionary into a list
    df_cell_gini["Gini"] = df_cell_gini["Gini"].apply(lambda x: list(x.values()))
    df_cell_gini = df_cell_gini["Gini"].apply(pd.Series)
    #return to a column names of pairs
    df_cell_gini.columns = all_sources
    df_cell_gini.index = df_cell_gini.index + 1

    return df_cell_gini





def str_to_np_array(s):
    # Remove unnecessary characters and split the string into lists of floats
    list_of_lists = [[float(num) for num in sublist.split()] for sublist in s[2:-2].replace('\n', '').split('] [')]
    # Convert list of lists to NumPy array
    return np.array(list_of_lists)

def str_to_dict(input_str):

    result_dict = ast.literal_eval(input_str)
    return result_dict
