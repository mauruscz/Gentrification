from collections import defaultdict
import mesa
import numpy as np
from .utils import *
from scipy.stats import percentileofscore
import random
from math import sqrt

class GentAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model, tipo, wealth):
        super().__init__(unique_id, model)

        self.tipo = tipo
        self.wealth = wealth


    def step(self):

        self.source = self.pos #(x,y) of the agent 
        step = self.model.schedule.steps #step of the simulation

        self.new_position = (-1,-1)

        cell_list = [cell[1] for cell in self.model.grid.coord_iter()]

        #retrieve an array of the richnesses of the agent in my same cell
        agents_in_my_cell = [agent.wealth for agent in self.model.grid.get_cell_list_contents([self.pos])]
        
        self.my_percentile = percentileofscore(agents_in_my_cell, self.wealth, kind='weak')
        self.my_percent= self.my_percentile/100

        
        if self.tipo == 'C':   #if the agent is low-income


            if self.model.mode == "random":
                #flip a coin
                my_prob = 0.5

            else:
                my_prob = 1-(self.my_percent **(1/2))

            s = random.random()

            if s < my_prob:

                self.model.desire_to_move_C += 1
                cells_with_space = eligibles_cells_with_space(self.model, cell_list)


                if len(cells_with_space) != 0:

                    if self.model.mode == "randomdest" or self.model.mode == "random":
                        self.new_position = random.choice(cells_with_space)

                    else:


                        empty_cells = [cell for cell in cells_with_space if self.model.grid.is_cell_empty(cell)]

                        cells2percentile = calculate_arriving_percentile(self.model, self, cells_with_space) 

                        cells2percentile = {k:v/100 for k,v in cells2percentile.items()}
                        cells2percentile = {k:
                                            v**(1/2)
                                            for k,v in cells2percentile.items()}
                        

                                                
                        cells2percentile.update({cell: random.random() for cell in empty_cells})
                                          

                        #if cells2percentile is not empty and the weights are not all 0
                        if len(cells2percentile) != 0 and sum(cells2percentile.values()) > 0:
                            
                            self.new_position = random.choices(list(cells2percentile.keys()), weights=cells2percentile.values())[0]

                        else:
                            self.model.unhappy_C += 1
                
                else:
                    self.model.unhappy_C += 1
        





        elif self.tipo == 'B': #if the agent is middle-income


            if self.model.mode == "random":
                #flip a coin
                my_prob = 0.5

            else:
                my_prob = 4 *(self.my_percent - 0.5)**2

            s = random.random()

            if s < my_prob:

                self.model.desire_to_move_B += 1
                    
                cells_with_space = eligibles_cells_with_space(self.model, cell_list)

                if len(cells_with_space) != 0:


                    if self.model.mode == "random" or self.model.mode == "randomdest":
                        self.new_position = random.choice(cells_with_space)
                    
                    else:
                        
                        empty_cells = [cell for cell in cells_with_space if self.model.grid.is_cell_empty(cell)]
                        cells2percentile = calculate_arriving_percentile(self.model, self, cells_with_space)

                        cells2percentile = {k:v/100 for k,v in cells2percentile.items()}

                        # First, calculate the score and store it back in cells2percentile
                        cells2percentile = {k: 1 - (4 * ((v - 0.5)**2)) for k, v in cells2percentile.items()}

                        cells2percentile.update({cell: random.random() for cell in empty_cells})


                        if len(cells2percentile) != 0 and sum(cells2percentile.values()) > 0:
                            self.new_position = random.choices(list(cells2percentile.keys()), weights=list(cells2percentile.values()))[0]


                        else:
                            self.model.unhappy_B += 1
                
                else:
                    self.model.unhappy_B += 1













                 
            


        elif self.tipo == 'A': #if the agent is high-income

            if self.model.mode == "random":
                #flip a coin
                my_prob = 0.5

                s = random.random()

                if s < my_prob:
                    #pick a random cell
                    self.model.desire_to_move_A += 1
                    cells_with_space = eligibles_cells_with_space(self.model, cell_list)

                    if len(cells_with_space) != 0:
                        self.new_position = random.choice(cells_with_space)

                    else:
                        self.model.unhappy_A += 1

            else:


                if step >= self.model.h:

                    s = random.random()

                    if s < self.model.p_g:

                        self.model.desire_to_move_A += 1

                        if self.model.mode == "randomdest":
                            cells_with_space = eligibles_cells_with_space(self.model, cell_list)
                            if len(cells_with_space) != 0:
                                self.new_position = random.choice(cells_with_space)
                            else:
                                self.model.unhappy_A += 1

                        else:

                            #cells_last_h_list is an array of median richness matrices of the cells in the last h steps
                            cells_last_h_list = self.model.median_richness_matrix_array[step-self.model.h : step]


                            cells_median_richnesses_last_h = defaultdict(list)

                            #in cells_richness_series_last_h we have a dictionary in which the key is the cell and the value is a list of the median richness of the cell in the last h steps
                            for cells_richs in cells_last_h_list:
                                #cells_rich is the matrix of richness at the first of the h steps
                                for i in range(len(cells_richs)):
                                    for j in range(len(cells_richs[0])):
                                        cells_median_richnesses_last_h[(i,j)].append(cells_richs[i][j])



                            # calculate the average growth rate of the median richness of the cells in the last h steps
                            cells_growth_rates = {cell: average_growth_rate(richness_series) for cell, richness_series in cells_median_richnesses_last_h.items()}


                            cells_with_space = eligibles_cells_with_space(self.model, cell_list)

                            
                            cells_probabilities = {cell: 
                                                            #temporal vision
                                                            cells_growth_rates[cell]

                                                            for cell in cells_with_space

                                                            if cells_growth_rates[cell] > cells_growth_rates[self.source] 
                                                            and cells_growth_rates[cell] > 0
                                                            }
                            
                            if len(cells_probabilities) != 0:
                                #self.new_position = self.model.random.choices(list(cells_probabilities.keys()), weights=list(cells_probabilities.values()))[0]
                                self.new_position = random.choices(list(cells_probabilities.keys()), weights=list(cells_probabilities.values()))[0]
                            else:
                                self.model.unhappy_A += 1


        if self.new_position != (-1,-1):
            self.model.grid.move_agent(self, self.new_position)