from types import SimpleNamespace
import numpy as np 
from scipy import optimize
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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


    """
    1: Edgeworth Box
    """
    #We define a function to calculate the indifference curves for our two consumers
    #First, We define the method
    def find_indifference_curve(self, consumer, w1, w2, alpha, N, x2_max):

        # We use this code block to select the appropriate utility function given the consumer (A or B)
        if consumer == 'A':
            utility_function = self.utility_A
        elif consumer == 'B':
            utility_function = self.utility_B
        else:
            raise ValueError("Consumer must be 'A' or 'B'.")

        # Get the current utility level at the initial endowment point from where we want to draw the indifference curves
        u = utility_function(w1, w2)

        # Initialize vectors for x1 and x2
        x1_vec = np.empty(N)
        x2_vec = np.linspace(1e-8, x2_max, N)

        # This is our root-finding loop. For each quantity of good 2, the loop calculates the corresponding quantity
        # of good 1 giving the same level of utility as in the endowment point
        for i, x2 in enumerate(x2_vec):
            def local_objective(x1):
                return utility_function(x1, x2) - u

            x1_guess = 0  # initial guess for root-finding
            result = optimize.root(local_objective, x1_guess)
            x1_vec[i] = result.x[0] if result.success else np.nan

        return x1_vec, x2_vec

# This code is used to draw our edgeworth box.
    def plot_edgeworth_box(self, N=75, x2_max=1):
        par = self.par
        # Calculate indifference curves for A and B
        x1A_vec, x2A_vec = self.find_indifference_curve('A', self.par.w1A, self.par.w2A, self.par.alpha, N, x2_max)
        x1B_vec, x2B_vec = self.find_indifference_curve('B', self.par.w1B, self.par.w2B, self.par.beta, N, x2_max)
        
        # Invert B's indifference curve
        x1B_inverted = 1 - x1B_vec
        x2B_inverted = 1 - x2B_vec
        
        # a. total endowment
        w1bar = 1.0
        w2bar = 1.0

        # b. figure set up
        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # Plot endowments and indifference curves
        ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='endowment')
        ax_A.plot(x1A_vec, x2A_vec, color='blue', label='Indifference Curve A')
        ax_A.plot(x1B_inverted, x2B_inverted, color='red', label='Indifference Curve B')
        
        # limits
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.7,1.0));
        
        # Filling the area between the curves and ensuring the vectors are sorted by x1
        sorted_indices_A = np.argsort(x1A_vec)
        sorted_indices_B = np.argsort(x1B_inverted)

        x1A_sorted = x1A_vec[sorted_indices_A]
        x2A_sorted = x2A_vec[sorted_indices_A]
        x1B_sorted = x1B_inverted[sorted_indices_B]
        x2B_sorted = x2B_inverted[sorted_indices_B]

        # Ensure we only color between the curves (to the left of Endowment A's x1)
        to_fill = (x1A_sorted <= self.par.w1A) & (x1A_sorted <= x1B_sorted)

        # Fill the area between the curves
        ax_A.fill_betweenx(x2A_sorted, x1A_sorted, x1B_sorted, where=to_fill, color='grey', alpha=0.3)

        plt.show()

    """
    2: Market clearing errors
    """
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
    
    """
    3: Market clearing price
    """
    def market_clearing_price(self, P1):
        """Find the market clearing price using a solver"""
        
        # Define a function to minimize - sum of absolute errors
        def objective_function(p1):
            eps1, eps2 = self.check_market_clearing(p1)
            return abs(eps1) + abs(eps2)

        # Find the market clearing price that minimizes the objective function
        result = minimize_scalar(objective_function, bounds=(min(P1), max(P1)), method='bounded')

        return result.x, result.fun

    """
    4a and 4b: A chooses the price to maximize her own utility.
    """
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
        allocation_A = (1 - x1B, 1 - x2B)
        allocation_B = (x1B, x2B)

        return max_utility_price, allocation_A, allocation_B, max_utility

    '''
    5a. Allocation in choice set C with A as the market maker.
    '''
    def allocation_within_choice_set(self, N=75):
        # Calculate initial endowments for A and B
        w1A, w2A = self.par.w1A, self.par.w2A
        w1B, w2B = self.par.w1B, self.par.w2B

        # Calculate utility of initial endowments for A and B
        uA_initial = self.utility_A(w1A, w2A)
        uB_initial = self.utility_B(w1B, w2B)

        # Create arrays of possible consumption levels for A and B
        xA_levels = np.linspace(0, 1, N + 1)
        xB_levels = 1 - xA_levels

        # Initialize variables to store the optimal allocation and maximum utility for A
        optimal_allocation_A_5a = (0, 0)
        max_utility_A_5a = 0

        # Iterate over all possible combinations of consumption levels for A and B
        for xA1 in xA_levels:
            for xA2 in xA_levels:
                # Calculate consumption levels for B
                xB1 = 1 - xA1
                xB2 = 1 - xA2

                # Calculate utility for A and B
                uA = self.utility_A(xA1, xA2)
                uB = self.utility_B(xB1, xB2)

                # Check if consumption levels satisfy the choice set conditions
                if uA >= uA_initial and uB >= uB_initial:
                    # Update optimal allocation and maximum utility for A if current allocation is better
                    if uA > max_utility_A_5a:
                        max_utility_A_5a = uA
                        optimal_allocation_A_5a = (xA1, xA2)

        # Calculate allocation for B corresponding to optimal allocation for A
        optimal_allocation_B_5a = (1 - optimal_allocation_A_5a[0], 1 - optimal_allocation_A_5a[1])

        return optimal_allocation_A_5a, optimal_allocation_B_5a, max_utility_A_5a

    '''
    5b. Allocation without restrictions, and still A as the market maker.
    '''
    def allocation_without_restrictions(self):
        # Define the objective function for A's utility
        def objective_function(x):
            xA1, xA2 = x
            return -self.utility_A(xA1, xA2)

        # Define the inequality constraint for B's utility
        def constraint_function(x):
            xB1, xB2 = 1 - x[0], 1 - x[1]
            return self.utility_B(xB1, xB2) - self.utility_B(self.par.w1B, self.par.w2B)

        # Define bounds for xA1 and xA2
        bounds = [(0, 1), (0, 1)]

        # Solve the optimization problem
        result = optimize.minimize(objective_function, x0=(0.5, 0.5), bounds=bounds, constraints={'type': 'ineq', 'fun': constraint_function})

        # Extract optimal allocation and maximum utility for A
        optimal_allocation_A = result.x
        max_utility_A = -result.fun

        # Calculate consumption levels for B corresponding to optimal allocation for A
        optimal_allocation_B = (1 - optimal_allocation_A[0], 1 - optimal_allocation_A[1])

        return optimal_allocation_A, optimal_allocation_B, max_utility_A

    '''
    6a. Allocation by the utilitarian social planner.
    '''
    def utilitarian_allocation(self):
        # Define the objective function as the negative of the sum of utilities
        def objective_function(x):
            xA1, xA2 = x
            return -(self.utility_A(xA1, xA2) + self.utility_B(1 - xA1, 1 - xA2))

        # Define bounds for xA1 and xA2
        bounds = [(0, 1), (0, 1)]

        # Solve the optimization problem
        result = optimize.minimize(objective_function, x0=(0.5, 0.5), bounds=bounds)

        # Extract optimal allocation and maximum aggregate utility
        optimal_allocation_A = result.x
        max_utility_aggregate = -result.fun

        # Calculate consumption levels for B corresponding to optimal allocation for A
        optimal_allocation_B = (1 - optimal_allocation_A[0], 1 - optimal_allocation_A[1])

        return optimal_allocation_A, optimal_allocation_B, max_utility_aggregate

    '''
    #FRENOR: Er det gammel kode der skal slettes?
    '''
    def calculate_allocation(self, x1A, x2A):
        x1B = 1 - x1A
        x2B = 1 - x2A

        return x1A, x1B, x2A, x2B