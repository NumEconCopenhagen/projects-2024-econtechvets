from types import SimpleNamespace
import numpy as np 

class InauguralProjectClass: 

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

        # c. total endowments
        par.w1B = 1-par.w1A
        par.w2B = 1-par.w2A

        # d. prices
        par.p2 = 1

    def utility_A(self,x1A,x2A):
        par = self.par
        return x1A**par.alpha*x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        par = self.par
        return x1B**par.beta*x2B**(1-par.beta)

    def demand_A(self,p1): # demand for good 1 by agent A, due to Walras law, we only need to find the demand for good 1
        par = self.par
        return par.alpha*(p1*par.w1A+par.p2*par.w2A)/p1

    def demand_B(self,p1): # demand for good 1 by agent B, due to Walras law, we only need to find the demand for good 1
        par = self.par
        return par.beta*(p1*par.w1B+par.p2*par.w2B)/p1

    def check_market_clearing(self,p1):

        par = self.par

        x1A = self.demand_A(p1)
        x1B = self.demand_B(p1)
        x2A = self.demand_A(par.p2)
        x2B = self.demand_B(par.p2)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    def calculate_market_clearing_errors(self, P1): #market clearing errors
        errors = []

        for p1 in P1:
            eps1, eps2 = self.check_market_clearing(p1)
            error = abs(eps1) + abs(eps2)
            errors.append(error)

        return errors
    
    def find_market_clearing_price(self, P1): # Finding the market clearing price that minimizes the sum of the market clearing errors 
        min_error = float('inf')
        market_clearing_price = None

        for p1 in P1: #Iterates over each p1 value in the set P1 and checks the market clearing errors
            eps1, eps2 = self.check_market_clearing(p1)
            error = abs(eps1) + abs(eps2)

            if error < min_error: #If the error is smaller than the current minimum error, the error becomes the new minimum error and the market clearing price becomes the new p1 value
                min_error = error
                market_clearing_price = p1

        return market_clearing_price #Returns the market clearing price