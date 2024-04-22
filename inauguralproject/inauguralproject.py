from types import SimpleNamespace
import numpy as np 
from scipy import optimize
from scipy.optimize import minimize_scalar

class InauguralProjectClass: 

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1-par.w1A
        par.w2B = 1-par.w2A

    def utility_A(self,x1A,x2A):
        """utility of consumer A"""
        par = self.par
        return x1A**par.alpha*x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        """utility of consumer B"""
        par = self.par
        return x1B**par.beta*x2B**(1-par.beta)

    def demand_A(self,p1, p2=1):
        """demand of consumer A"""
        par = self.par
        x1A = par.alpha*(p1*par.w1A+p2*par.w2A)/p1
        x2A = (1-par.alpha)*(p1*par.w1A+p2*par.w2A)/p2
        return x1A, x2A


    def demand_B(self,p1, p2=1):
         """demand of consumer B"""
         par = self.par
         x1B = par.beta*(p1*par.w1B+p2*par.w2B)/p1
         x2B = (1-par.beta)*(p1*par.w1B+p2*par.w2B)/p2
         return x1B, x2B

    def utility_A_with_price(self, p1, w1B, w2B):
        """Utility of consumer A with given prices and endowments of B"""
        par = self.par
        x1B, x2B = self.demand_B(p1)
        return self.utility_A(1 - x1B, 1 - x2B)

    def max_utility_A_allocation(self, P1):
        """Find the allocation maximizing consumer A's utility"""
        def negative_utility(p1):
            return -self.utility_A_with_price(p1, self.par.w1B, self.par.w2B)

        result = minimize_scalar(negative_utility, bounds=(min(P1), max(P1)), method='bounded')
        max_utility_price = result.x
        max_utility = -result.fun
        x1B, x2B = self.demand_B(max_utility_price)
        allocation = (1 - x1B, 1 - x2B)

        return max_utility_price, allocation, max_utility

    def check_market_clearing(self,p1):
        par = self.par
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)
        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)
        return eps1,eps2
    
    def calculate_market_clearing_errors(self, P1): 
        """Calculate market clearing errors for a given set of prices P1"""
        errors_1 = []
        errors_2 = []

        for p1 in P1:
            eps1, eps2 = self.check_market_clearing(p1)
            errors_1.append(eps1)
            errors_2.append(eps2)

        return errors_1, errors_2
    
    def market_clearing_price(self, P1):
        """Find the market clearing price using a solver"""
        
        # Define a function to minimize - sum of absolute errors
        def objective_function(p1):
            eps1, eps2 = self.check_market_clearing(p1)
            return abs(eps1) + abs(eps2)

        # Find the market clearing price that minimizes the objective function
        result = minimize_scalar(objective_function, bounds=(min(P1), max(P1)), method='bounded')

        return result.x, result.fun    

    def maximize_aggregate_utility(self):
        max_utility = float('-inf')
        optimal_x1A = None
        optimal_x2A = None

        for x1A in np.linspace(0, 1, 101):  # 101 points between 0 and 1
            for x2A in np.linspace(0, 1, 101):
                utility = self.utility_A(x1A, x2A) + self.utility_B(1 - x1A, 1 - x2A)
                if utility > max_utility:
                    max_utility = utility
                    optimal_x1A = x1A
                    optimal_x2A = x2A

        return optimal_x1A, optimal_x2A

    def calculate_allocation(self, x1A, x2A):
        x1B = 1 - x1A
        x2B = 1 - x2A

        return x1A, x1B, x2A, x2B