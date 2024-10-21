from collections import defaultdict
import mesa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .agent import GentAgent
from .utils import *

import sys

    


class GentModel(mesa.Model):
    """Modelling Gentrification in a lattice city | Author: Giovanni Mauro, 2023"""

    def __init__(self,  mode = "improve", starting_deployment="centre_segr", width = 7, height = 7, num_agents=5000, h = 10, 
                 p_g = 0.1, empty_border = 0, tqdm_on = True, termination_condition = True,
                  seed = 67):

        self.seed = seed
        self.num_agents = num_agents
        self.mode = mode
        self.tqdm_on = tqdm_on  
        self.termination_condition = termination_condition
        self.starting_deployment = starting_deployment
        self.h = h
        self.p_g = p_g
        self.width = width
        self.height = height
        self.empty_border = empty_border

        if self.mode not in ['random', 'randomdest', 'improve']:
            print("Error: mode must be either 'random', 'randomdest', or 'improve'")
            sys.exit()


        if self.empty_border != 0:
            self.width = self.width + 2*self.empty_border
            self.height = self.height + 2*self.empty_border


        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)

        self.starting_deployment = starting_deployment





        self.df = pd.read_csv("income/income_clean.csv")


        # ------ Capacity matrix ------

        capacity_number = calculate_neighborhood_capacity(num_agents, self.width*self.height, occupancy_rate=0.75)
        self.capacity = np.zeros((self.width, self.height))
        for cell in self.grid.coord_iter():
            x = cell[1][0]
            y = cell[1][1]
            self.capacity[x][y] = capacity_number


        # Create agents
        for i in range(self.num_agents):
          
            
            row = pick_random_row(df = self.df, model = self, percent_cumul_limit_high=99.99911)
            if row<=5: #if the income is up to 6th row
                tipo = "C"

            elif row>5 and row<=35:
                tipo = "B"

            elif row>35:
                tipo = "A"


            income = pick_random_amount(self.df, row, self)   

            #place agent according to starting_deployment mode
            (x,y) = place_agent_2(self, tipo)


            a = GentAgent(i, self, tipo = tipo, wealth=income)
            self.grid.place_agent(a, (x,y))
            self.schedule.add(a)

                








# ------ Averages (median mean std) richness matrix inizialization ------
            

        self.median_richness_matrix_array = []
        self.median_richness_matrix = calculate_median_richness_matrix(self)
        self.median_richness_matrix_array.append(self.median_richness_matrix)


        self.presence_matrix = np.zeros((self.width, self.height))
        for agent in self.schedule.agents:
            self.presence_matrix[agent.pos] += 1



        self.cells2wealths = defaultdict(list)
        for agent in self.schedule.agents:
            self.cells2wealths[agent.pos].append(agent.wealth)

        self.cells2types = defaultdict(list)
        for agent in self.schedule.agents:
            self.cells2types[agent.pos].append(agent.tipo)

        
        self.total_richness_matrix_array = []
        self.total_richness_matrix = calculate_total_richness_matrix(self)
        self.total_richness_matrix_array.append(self.total_richness_matrix)

 

        self.datacollector = mesa.DataCollector(
            agent_reporters={"pos": lambda a: a.pos,
                              "tipo": lambda a: a.tipo, 
                             "new_position": lambda a: a.new_position, 
                             "source": lambda a: a.source},
            model_reporters={
                                "median_richness_matrix": lambda _: _.median_richness_matrix,
                             }
        )



        self.running = True


        self.continuos_unhappy_C = 0


    def step(self):


        self.unhappy_A = 0
        self.unhappy_B = 0
        self.unhappy_C = 0

        self.desire_to_move_C = 0
        self.desire_to_move_B = 0
        self.desire_to_move_A = 0

        self.schedule.step()

        self.mean_richness_matrix = calculate_mean_richness_matrix(self)
        self.mean_richness_matrix_array.append(self.mean_richness_matrix)




        self.median_richness_matrix = calculate_median_richness_matrix(self)
        self.median_richness_matrix_array.append(self.median_richness_matrix)


        self.std_dev_richness_matrix = calculate_std_dev_richness_matrix(self)
        self.std_dev_richness_matrix_array.append(self.std_dev_richness_matrix)

        self.total_richness_matrix = calculate_total_richness_matrix(self)
        self.total_richness_matrix_array.append(self.total_richness_matrix)

        self.presence_matrix = np.zeros((self.width, self.height))
        for agent in self.schedule.agents:
            self.presence_matrix[agent.pos] += 1

        self.cells2wealths = calculate_cells2wealths(self)
        self.cells2types = calculate_cells2types(self)


        
        self.datacollector.collect(self)



        if self.termination_condition:
            #print("unhappy C: ", self.unhappy_C, " desire to move C: ", self.desire_to_move_C)
            if self.unhappy_C == self.desire_to_move_C:
                #print("Gentrification process has ended")
                self.running = False




    def run_model(self, n):

        if self.tqdm_on:
            for i in tqdm(range(n)):
                if self.running == True:
                    self.step()
                else:
                    break
        else:
            for i in range(n):
                if self.running == True:
                    self.step()
                else:
                    break            


