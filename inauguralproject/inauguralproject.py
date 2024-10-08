from types import SimpleNamespace
import numpy as np 
from scipy import optimize
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
    def find_indifference_curve(self, consumer, w1, w2, alpha, N, x2_max):
        """
        Calculates the indifference curve for a given consumer, starting from their endowment and covering a range of values for one of the goods.

        Args:
            consumer (str): The consumer for whom to calculate the curve ('A' or 'B').
            w1 (float): Initial endowment of good 1 for the consumer.
            w2 (float): Initial endowment of good 2 for the consumer.
            alpha (float): The weight (preference) given to good 1 in the consumer's utility function.
            N (int): Number of points to calculate along the indifference curve.
            x2_max (float): The maximum value for good 2 to use in generating the curve.

        Returns:
            tuple (numpy.ndarray, numpy.ndarray): Two arrays, one for the quantity of good 1 and one for good 2, representing points along the indifference curve.

        This method selects the appropriate utility function for the specified consumer to compute the indifference curve. 
        """
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
    def plot_edgeworth_box(self, N=75, x2_max=1, comparison_data=None):
        """
        Plots an Edgeworth Box illustrating the allocations of two goods between two consumers, A and B.

        Args:
            allocation_A: Allocation for consumer A, specified as a tuple (x1A, x2A), where x1A is the amount of good 1 and x2A is the amount of good 2.
            allocation_B: Allocation for consumer B, specified as a tuple (x1B, x2B), where x1B is the amount of good 1 and x2B is the amount of good 2.

        Returns:
            None: This function does not return any value but displays a matplotlib plot showing the Edgeworth Box.

        This method visualizes the consumption allocations of two goods between two consumers within an Edgeworth Box framework.
        """
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
        ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='Initial endowment')
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

        # Add comparison data points if provided
        if comparison_data:
            for label, coords in comparison_data.items():
                ax_A.scatter(coords[0], coords[1], label=label, alpha=0.7)

        # Legend
        ax_A.legend(frameon=True, loc='upper right', bbox_to_anchor=(1.7, 1.0))

        plt.show()

    """
    2: Market clearing errors
    """
    # First, defining the set P1
    def generate_set_P1(self, start, end, N):
        """
        Generates and returns set P1, a collection of points representing possible allocations of goods.

        Args:
            num_points (int): The number of points to generate in set P1.

        Returns:
            numpy.ndarray: An array of points, each represented by a tuple of coordinates, forming the set P1.

        This method generates a set of points representing feasible allocations.
        """
        # Start at 0.5 and add (2*i/N) for each term
        P1 = [start + (2*i)/N for i in range(N)]
        # Ensure that the last element is exactly 2.5
        P1[-1] = end
        return P1

    # We define a method to check the market clearing
    def check_market_clearing(self,p1):
        """
        Checks if the given allocations result in market clearing

        Args:
            allocations (dict): A dictionary containing the allocations for consumers

        Returns:
            bool: Returns True if the market clears (supply equals demand for all goods), otherwise False.

        This method evaluates whether the sum of goods allocated to all consumers matches the total available supply. It is used in the next method to calculate the market clearing errors.
        """
        par = self.par
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)
        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)
        return eps1,eps2
    
    # We define a method to calculate the market clearing errors
    def calculate_market_clearing_errors(self, P1): 
        """Calculate market clearing errors for a given set of prices P1 using the methods defined above"""
        errors_1 = []
        errors_2 = []

        for p1 in P1:
            eps1, eps2 = self.check_market_clearing(p1)
            errors_1.append(eps1)
            errors_2.append(eps2)

        return errors_1, errors_2
    
    # We define a method to plot the market clearing errors
    def plot_market_clearing_errors(self, P1_set):
        """
        Plots the market clearing errors for a given set of prices P1_set.

        Parameters:
        P1_set (list): A list of prices at which to calculate the market clearing errors.
        """
        # Calculate errors using the new set of prices
        error_1, error_2 = self.calculate_market_clearing_errors(P1_set)

        # Plotting errors
        plt.figure(figsize=(10, 5))
        plt.plot(P1_set, error_1, label='Error in market clearing for x1')
        plt.plot(P1_set, error_2, label='Error in market clearing for x2')
        plt.xlabel('Price (p1)')
        plt.ylabel('Error in market clearing')
        plt.title('Errors in Market Clearing with P1_set')
        plt.grid(True)
        plt.legend()
        plt.show()
    

    """
    3: Market clearing price
    """
    def find_market_clearing_price(self):
        """
        Determines the market clearing price for goods in the exchange economy

        Args:
            initial_price_guess (float): An initial guess for the price to start the optimization process.

        Returns:
            float: The market clearing price, at which demand equals supply.

        This method utilizes numerical optimization techniques to find the price that equates the total demand and supply of goods within the market.
        """
        # Objective function: sum of absolute values of market-clearing errors
        objective_function = lambda p1: abs(self.check_market_clearing(p1)[0]) + abs(self.check_market_clearing(p1)[1])

        # Lower bound set slightly above zero to avoid division by zero
        lower_bound = 1e-8

        # Use a solver to find the root of the objective function
        result = minimize_scalar(objective_function, bounds=(lower_bound, 10), method='bounded')
        if result.success:
            return result.x
        else:
            raise ValueError("Solver did not converge.")

    """
    4a: A chooses the price in P1 to maximize her own utility.
    """
    # We define the method to find the market clearing price using the set P1
    def max_utility_A(self, P1_set):
        """
        Maximizes the utility of consumer A by optimizing their allocation of goods for a price in P1_set.

        Args:
            None

        Returns:
            tuple (float, float): The optimal allocation of goods for consumer A, given as (x1A, x2A).

        This method finds the combination of goods 1 and 2 that maximizes the utility for consumer A for a price within P1_set.
        """
        max_utility = -np.inf
        optimal_price = None
        optimal_allocation_A = None
        optimal_allocation_B = None

        for p1 in P1_set:
            # Calculate consumer B's demand at the given price
            x1B, x2B = self.demand_B(p1)

            # Make sure the allocations are non-negative
            if x1B < 0 or x1B > 1 or x2B < 0 or x2B > 1:
                continue  # Skip this iteration if the allocation is negative or greater than total endowment

            x1A = 1 - x1B
            x2A = 1 - x2B

            # Calculate utility ensuring non-negative quantities
            if x1A >= 0 and x2A >= 0:
                utility_A = self.utility_A(x1A, x2A)
                if utility_A.real > max_utility:  # Compare only the real part of utility_A
                    max_utility = utility_A.real
                    optimal_price = p1
                    optimal_allocation_A = (x1A, x2A)
                    optimal_allocation_B = (x1B, x2B)

        return optimal_price, optimal_allocation_A, optimal_allocation_B

    """
    4b: A chooses any positive price to maximize her own utility.
    """ 
    def max_utility_A_continuous(self):
        """
        Maximizes the utility of consumer A across a continuous range of good allocations (not restricted by P1_set like above)

        Args:
            bounds (tuple of (float, float)): The bounds within which to optimize the allocations of goods for consumer A.

        Returns:
            tuple (float, float): The optimal allocation of goods for consumer A that maximizes their utility, represented as (x1A, x2A).

        This method employs a continuous optimization approach to find the allocation of goods that maximizes consumer A's utility. The method is not restricted by the discrete set P1_set.
        """
        # Define the negative of the utility function of A as the objective
        def objective(p1):
            # Calculate consumer B's demand at the given price
            x1B, x2B = self.demand_B(p1)

            # Compute the remaining allocations for A
            x1A = 1 - x1B
            x2A = 1 - x2B

            # Return the negative of A's utility for minimization (if allocations are non-negative)
            if x1A >= 0 and x2A >= 0:
                return -self.utility_A(x1A, x2A).real
            else:
                # Return infinity if allocations are negative, so the optimizer avoids these values
                return np.inf

        # Set the initial guess and bounds for the price p1
        initial_guess = [1.0]
        bounds = [(1e-8, None)]  # p1 should be positive, and we set a lower bound to avoid division by zero

        # Use an optimizer to find the price that minimizes the objective function
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

        # If the optimization is successful, extract the optimal price and compute the allocations
        if result.success:
            optimal_price_continuous = result.x[0]
            x1B, x2B = self.demand_B(optimal_price_continuous)
            optimal_allocation_A_continuous = (1 - x1B, 1 - x2B)
            optimal_allocation_B_continuous = (x1B, x2B)
            return optimal_price_continuous, optimal_allocation_A_continuous, optimal_allocation_B_continuous
        else:
            raise ValueError("Optimization failed to converge.")

    '''
    5a. Allocation in choice set C with A as the market maker.
    '''
    def pareto_optimal_allocations(self):
        """
        Identifies and returns Pareto optimal allocations between our two consumers A and B.

        Args:
            None

        Returns:
            list of tuples: A list of Pareto optimal allocations, where each tuple (x1A, x2A, x1B, x2B) represents the allocation of goods between the two consumers.

        This method calculates and lists all allocations that are Pareto optimal within the model's constraints. 
        """
        # Calculate initial utilities for A and B
        utility_A_initial = self.utility_A(self.par.w1A, self.par.w2A)
        utility_B_initial = self.utility_B(self.par.w1B, self.par.w2B)

        # Initialize variables for the optimal allocation
        max_utility_A = -np.inf
        optimal_allocation_A = None
        optimal_utility_A = None
        optimal_utility_B = None
        
        N = 75

        # Iterate over the discrete grid of allocations for A
        for x1A in [i/N for i in range(N+1)]:
            for x2A in [i/N for i in range(N+1)]:
                x1B = 1 - x1A
                x2B = 1 - x2A
                
                # Ensure B is at least as well off as in the initial endowment
                utility_B = self.utility_B(x1B, x2B)
                if utility_B >= utility_B_initial:
                    # Check if this allocation provides a higher utility for A than the current max
                    utility_A = self.utility_A(x1A, x2A)
                    if utility_A > max_utility_A:
                        max_utility_A = utility_A
                        optimal_allocation_A = (x1A, x2A)
                        optimal_utility_A = utility_A
                        optimal_utility_B = utility_B
        
        return optimal_allocation_A, optimal_utility_A, optimal_utility_B, utility_A_initial, utility_B_initial

    '''
    5b. Allocation if no further restrictions are imposed and with A as the market maker.
    '''
    def pareto_optimal_allocations_5b(self):
        """
        Identifies and returns Pareto optimal allocations for our two consumers

        Args:
            None

        Returns:
            list of tuples: A list of Pareto optimal allocations, where each tuple (x1A, x2A, x1B, x2B) represents the specific allocation of goods.

        This method calculates and list all Pareto optimal allocations. 
        """
        # Calculate initial utilities for A and B
        utility_A_initial = self.utility_A(self.par.w1A, self.par.w2A)
        utility_B_initial = self.utility_B(self.par.w1B, self.par.w2B)

        # Define the objective function: maximize utility of A (minimize negative utility)
        def objective(x):
            x1A, x2A = x[0], x[1]
            return -self.utility_A(x1A, x2A)  # Negative because minimize function is used

        # Constraints to ensure B is not worse off
        def constraint(x):
            x1A, x2A = x[0], x[1]
            x1B = 1 - x1A
            x2B = 1 - x2A
            return self.utility_B(x1B, x2B) - utility_B_initial

        # Bounds for x1A and x2A (can't be negative or exceed total resources)
        bounds = ((0, 1), (0, 1))

        # Initial guess (could be the initial endowment of A)
        initial_guess = [self.par.w1A, self.par.w2A]

        # Define the constraint as a dictionary
        cons = {'type': 'ineq', 'fun': constraint}

        # Perform the optimization
        result = minimize(objective, initial_guess, bounds=bounds, constraints=cons)

        if result.success:
            optimal_x1A, optimal_x2A = result.x
            optimal_utility_A_5b = self.utility_A(optimal_x1A, optimal_x2A)
            optimal_x1B = 1 - optimal_x1A
            optimal_x2B = 1 - optimal_x2A
            optimal_utility_B_5b = self.utility_B(optimal_x1B, optimal_x2B)
            return (optimal_x1A, optimal_x2A), optimal_utility_A_5b, optimal_utility_B_5b, utility_A_initial, utility_B_initial
        else:
            raise Exception("Optimization failed.")
    
    '''
    6a. Allocation by the utilitarian social planner.
    '''
    def utilitarian_allocation(self):
        """
        Method to solve the model seen from a social planner's perspective, maximizing the sum of utilities for consumers A and B.

        Args:
            None

        Returns:
            tuple (float, float, float): The optimal allocation for consumer A (xA1, xA2) and the maximum aggregate utility achieved by this allocation.

        This method uses a numerical optimization technique to maximize the sum of utilities across all consumers, achieving the utilitarian optimal allocation. 
        """
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
    7. Illustrating the results
    '''
    def plot_random_set_W(self, num_elements=50, seed=123):
        # Set the seed for reproducibility
        np.random.seed(seed)

        # Generate random elements for set W
        W = np.random.rand(num_elements, 2)

        # Plot set W
        plt.figure(figsize=(8, 6))
        plt.plot(W[:, 0], W[:, 1], 'bo', label='Elements of set W')
        plt.xlabel('w1A')
        plt.ylabel('w2A')
        plt.title(f'Set W with {num_elements} Elements')
        plt.grid(True)
        plt.legend()
        plt.show()

    '''
    8. Market equilibria for random draws
    '''
    def find_market_equilibrium_allocation(self):
        """
        Finds the market equilibrium allocation of goods.

        Returns:
            tuple: The equilibrium allocations for consumers A and B.
        """
        # Use the find_market_clearing_price method to find the equilibrium price
        p1_eq = self.find_market_clearing_price()

        # Calculate the equilibrium allocations for consumers A and B
        x1A_eq, x2A_eq = self.demand_A(p1_eq)
        x1B_eq, x2B_eq = self.demand_B(p1_eq)

        return x1A_eq, x2A_eq, x1B_eq, x2B_eq

    def save_state(self):
        """Saves the current state of the class parameters."""
        self.saved_state = (self.par.w1A, self.par.w2A, self.par.w1B, self.par.w2B)
    
    def restore_state(self):
        """Restores the class parameters to their saved state."""
        self.par.w1A, self.par.w2A, self.par.w1B, self.par.w2B = self.saved_state

    def find_and_plot_equilibria_for_random_draws(self, num_draws, seed=123):
        np.random.seed(seed)
        random_draws = np.random.rand(num_draws, 2)
        comparison_data = {}

        # Save the current state before the loop
        self.save_state()

        for i, draw in enumerate(random_draws):
            # Update the class parameters for each draw
            self.par.w1A, self.par.w2A = draw[0], draw[1]
            self.par.w1B, self.par.w2B = 1 - draw[0], 1 - draw[1]
            
            # Find the equilibrium allocations for the current draw
            x1A_eq, x2A_eq, x1B_eq, x2B_eq = self.find_market_equilibrium_allocation()
            comparison_data[f'Draw {i+1}'] = (x1A_eq, x2A_eq)

            # Restore the original state after each draw
            self.restore_state()

        # Now plot the Edgeworth box with the calculated equilibria
        self.plot_edgeworth_box(comparison_data=comparison_data)