from types import SimpleNamespace
import numpy as np
from scipy import optimize

class ProductionEconomyClass():

    def __init__(self):

        par = self.par = SimpleNamespace()
        ## a. parameters
        # firms
        par.A = 1.0
        par.gamma = 0.5

        # households
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0

        # government
        par.tau = 0.0
        par.T = 0.0

        # b. grids
        par.num_p1 = 10
        par.grid_p1 = np.linspace(0.1,2.0,par.num_p1)
        par.grid_mkt_clearing = np.zeros(par.num_p1)

        # Question 3
        par.kappa = 0.1