import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar, minimize

class ProductionEconomy:
    """
    A class to model the economy given in the exam question, problem 1. We provide the class with relevant parameter values and methods
    to calculate the firm and consumer behaviors. The class also includes methods to plot the results of the model.
    
    Attributes:
        A: Technology parameter influencing production efficiency.
        gamma: Returns to labor parameter in the production function.
        alpha: Preference weight in consumption.
        nu: Disutility of labor coefficient.
        epsilon: Elasticity parameter for labor supply.
        tau: Tax
        T: Lump-sum transfer
        w: Wage rate
        p1: Price of good 1
        p2: Price of good 2
    """
    def __init__(self):
        self.A = 1.0
        self.gamma = 0.5
        self.alpha = 0.3
        self.nu = 1.0
        self.epsilon = 2.0
        self.tau = 0.0
        self.T = 0.0
        self.w = 1.0
        self.p1 = 1.0  # setting the price of good 1 to an arbitrary value (default value)
        self.p2 = 1.0  # setting the price of good 2 to an arbitrary value (default value)
        self.kappa = 0.1  # Arbitrary value for kappa, can be adjusted

    def set_prices(self, p1, p2):
        """
        Sets the market prices for goods 1 and 2.
        Parameters:
            p1: Price of good 1.
            p2: Price of good 2.
        """
        self.p1 = p1
        self.p2 = p2

    def optimal_labor_demand(self, p, w):
        """
        Computes the optimal labor demand according to the function given in the exam question incl. given price and wage.
        Parameters:
            p: Price of the good.
            w: Wage rate.
        Returns:
            Optimal labor demand given price and wage.
        """
        return ((p * self.A * self.gamma) / w) ** (1 / (1 - self.gamma))

    def optimal_production(self, l):
        """
        Calculates production output based on labor according to the function given in the exam question.
        Parameter:
            l: Labor input.
        Returns:
            Production output given labor.
        """
        return self.A * (l ** self.gamma)
    
    def optimal_profit(self, p, w):
        """
        Calculates profit for a firm given according to the function given in the exam question given price and wage.
        Parameters:
            p: Price of the good.
            w: Wage rate.
        Returns:
            Optimal profit given price and wage.
        """
        profit_star = (1 - self.gamma) / self.gamma * (p * (self.A * self.gamma) / w) ** (self.gamma / (1 - self.gamma))
        return profit_star

    def maximize_utility(self):
        """
        Maximizes the utility of the central agent using numerical optimization.
        Returns:
            Labor supply that maximizes utility given the prices of goods 1 and 2.
        """
        def objective(l):
            if l <= 0:
                return np.inf
            utility = self.utility_function(l)
            if np.isnan(utility) or np.isinf(utility):
                return np.inf
            return -utility

        initial_guess = 0.1  # Set a better initial guess to avoid zero labor supply
        result = minimize(objective, [initial_guess], bounds=[(1e-6, None)])  # Use a small positive lower bound
        if result.success and not np.isnan(result.x[0]) and not np.isinf(result.x[0]):
            return result.x[0]
        return 0.0  # Return 0.0 if optimization fails


    def utility_function(self, l):
        """
        Utility function as a function of labor supply.
        Parameter:
            l: Labor supplied by the agent.
        Returns:
            Utility value.
        """
        pi_1 = self.optimal_profit(self.p1, self.w)  # Firm 1 profit
        pi_2 = self.optimal_profit(self.p2, self.w)  # Firm 2 profit
        c1, c2 = self.consumer_behavior(l, pi_1, pi_2)  # Consumption of goods 1 and 2 given firm 1 and 2 profits
        if c1 <= 0 or c2 <= 0 or np.isnan(c1) or np.isnan(c2) or np.isinf(c1) or np.isinf(c2):
            print(f"Invalid consumption values for labor supply {l}: c1={c1}, c2={c2}")
            return -np.inf  # Return negative infinity if consumption is not valid
        utility = np.log(c1 ** self.alpha * c2 ** (1 - self.alpha)) - self.nu * (l ** (1 + self.epsilon) / (1 + self.epsilon))
        if np.isnan(utility) or np.isinf(utility):
            print(f"Invalid utility for labor supply {l}: {utility}")
        return utility


    def consumer_behavior(self, l, pi_1, pi_2):
        """
        Calculates the consumption of goods 1 and 2 given labor supply and profits of firms 1 and 2.
        Parameters:
            l: Labor supply.
            pi_1: Profit of firm 1.
            pi_2: Profit of firm 2.
        Returns:
            Consumption of goods 1 and 2.
        """
        epsilon = 1e-6  # Small positive value to avoid zero or negative consumption
        # Initial total income without T
        initial_total_income = self.w * l + pi_1 + pi_2
        # Calculate c2 first without considering T
        c2 = (1 - self.alpha) * initial_total_income / (self.p2 + self.tau)
        # Calculate T as tau * c2
        self.T = self.tau * c2
        # Recalculate total income including T
        total_income = self.w * l + self.T + pi_1 + pi_2
        # Calculate c1 and c2 with the updated total income
        c1 = self.alpha * total_income / self.p1
        c2 = (1 - self.alpha) * total_income / (self.p2 + self.tau)
        return max(c1, epsilon), max(c2, epsilon)


    def plot_labor_market_clearing(self, p1_range, p2_range):
        """
        Plots the labor market clearing condition by iterating over specified ranges for p1 and p2.
        Parameters:
            p1_range: A range of prices for good 1.
            p2_range: A range of prices for good 2.
        """
        total_labor_demand = []  # creating empty lists to store the results for labor demand, l1 + l2
        optimal_labor_supply = []  # creating empty lists to store the results for labor supply, l*

        for price1 in p1_range:  # iterating over the range of prices for good 1
            for price2 in p2_range:  # iterating over the range of prices for good 2
                self.set_prices(price1, price2) 
                optimal_l = self.maximize_utility() 
                if optimal_l is not None: 
                    optimal_l1 = self.optimal_labor_demand(price1, self.w)  # calculating optimal labor demand for good 1
                    optimal_l2 = self.optimal_labor_demand(price2, self.w)  # calculating optimal labor demand for good 2
                    total_labor_demand.append(optimal_l1 + optimal_l2)  # calculating total labor demand
                    optimal_labor_supply.append(optimal_l)  # calculating labor supply

        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.scatter(total_labor_demand, optimal_labor_supply, color='blue')
        plt.plot([0, max(total_labor_demand)], [0, max(total_labor_demand)], 'r--', label='Market Clearing Line')
        plt.xlabel('Total Labor Demand $(l_1^* + l_2^*)$')
        plt.ylabel('Optimal Labor Supply $(l^*)$')
        plt.title('Labor Market Clearing Condition')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_good1_market_clearing(self):
        """
        Plots the market clearing condition for good 1 by iterating over a range of prices for p1. prices horizontally. difference between production and consumption vertically.
        """
        p1 = np.linspace(0.1, 2, 10)  # creating a range of prices for good 1 given the specifications in the question
        price_values = []  # creating empty lists to store the results for price values
        differences = []  # creating empty lists to store the results for differences between production and consumption

        for price1 in p1:  # iterating over the range of prices for good 1
            self.set_prices(price1, p2=1.0)  # setting the price of good 1 according to the specifications. Choosing an arbitrary fixed price for good 2
            optimal_l = self.maximize_utility() 
            if optimal_l is not None:
                optimal_l1 = self.optimal_labor_demand(price1, self.w)  # calculating optimal labor demand for good 1
                prod_y1 = self.optimal_production(optimal_l1)  # calculating production output for good 1
                pi_1 = self.optimal_profit(price1, self.w)  # calculating profit for firm 1
                pi_2 = self.optimal_profit(1.0, self.w)  # calculating profit for firm 2 given arbitrary fixed price for good 2
                cons_c1, _ = self.consumer_behavior(optimal_l, pi_1, pi_2)  # calculating consumption of good 1

                price_values.append(price1)  # appending the price values to the list
                differences.append(prod_y1 - cons_c1)  # appending the differences between production and consumption to the list (if 0, we have market clearing)

        plt.figure(figsize=(10, 6))
        plt.plot(price_values, differences, marker='o', linestyle='-', color='blue')
        plt.axhline(0, color='red', linestyle='--', label='Market Clearing Line (y=c)')
        plt.xlabel('Price of Good 1 ($p_1$)')
        plt.ylabel('Difference between Production and Consumption ($y_1^* - c_1^*$)')
        plt.title('Market Clearing Condition for Good 1 Across Different Prices')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_good2_market_clearing(self):
        """
        Plots the market clearing condition for good 2 by iterating over a range of prices for p2. prices horizontally. difference between production and consumption vertically.
        """
        p2 = np.linspace(0.1, 2, 10)  # creating a range of prices for good 2 given the specifications in the question
        price_values = []  # creating empty lists to store the results for price values
        differences = []  # creating empty lists to store the results for differences between production and consumption

        for price2 in p2:  # iterating over the range of prices for good 2
            self.set_prices(self.p1, price2)  # setting the price of good 2 according to the specifications. Using an arbitrary fixed price for good 1
            optimal_l = self.maximize_utility() 
            if optimal_l is not None:
                optimal_l2 = self.optimal_labor_demand(price2, self.w)  # calculating optimal labor demand for good 2
                prod_y2 = self.optimal_production(optimal_l2)  # calculating production output for good 2
                pi_1 = self.optimal_profit(1.0, self.w)  # calculating profit for firm 1 given arbitrary fixed price for good 1
                pi_2 = self.optimal_profit(price2, self.w)  # calculating profit for firm 2
                _, cons_c2 = self.consumer_behavior(optimal_l, pi_1, pi_2)  # calculating consumption of good 2

                price_values.append(price2)  # appending the price values to the list
                differences.append(prod_y2 - cons_c2)  # appending the differences between production and consumption to the list (if 0, we have market clearing)

        plt.figure(figsize=(10, 6))
        plt.plot(price_values, differences, marker='o', linestyle='-', color='blue')
        plt.axhline(0, color='red', linestyle='--', label='Market Clearing Line (y = c)')
        plt.xlabel('Price of Good 2 ($p_2$)')
        plt.ylabel('Difference between Production and Consumption ($y_2^* - c_2^*$)')
        plt.title('Market Clearing Condition for Good 2 Across Different Prices')
        plt.legend()
        plt.grid(True)
        plt.show()

    def market_clearing_prices(self):
        """
        Calculates the equilibrium prices for goods 1 and 2 by solving the system of equations for market clearing.
        Returns:
            Equilibrium prices for goods 1 and 2.
        """
        def equations(p):
            p1, p2 = p 
            self.set_prices(p1, p2) 
            l_star = self.maximize_utility()  # calculating optimal labor supply using the method defined above

            if l_star is None or l_star <= 0 or np.isnan(l_star) or np.isinf(l_star):
                return [np.inf, np.inf]  # Return large values if optimization fails

            l1_star = self.optimal_labor_demand(p1, self.w)  # calculating optimal labor demand for good 1
            l2_star = self.optimal_labor_demand(p2, self.w)  # calculating optimal labor demand for good 2
            
            y1_star = self.optimal_production(l1_star)  # calculating production output for good 1
            y2_star = self.optimal_production(l2_star)  # calculating production output for good 2
            
            pi_1 = self.optimal_profit(p1, self.w)  # calculating profit for firm 1
            pi_2 = self.optimal_profit(p2, self.w)  # calculating profit for firm 2
            c1_star, c2_star = self.consumer_behavior(l_star, pi_1, pi_2)  # calculating consumption of goods 1 and 2
            
            if np.isnan(y1_star) or np.isnan(c1_star) or np.isnan(y2_star) or np.isnan(c2_star) or \
            np.isinf(y1_star) or np.isinf(c1_star) or np.isinf(y2_star) or np.isinf(c2_star):
                return [np.inf, np.inf]  # Return large values if any calculation results in NaN or inf

            # Return a flattened array of differences
            return [y1_star - c1_star, y2_star - c2_star]  # returning the differences between production and consumption for goods 1 and 2 that are also market clearing condition 1+2

        # Initial guesses for p1 and p2
        initial_guesses = [1.0, 1.0]  # setting initial guesses for prices of goods 1 and 2 (arbitrary values)
        equilibrium_prices = fsolve(equations, initial_guesses)  # solving the system of equations using fsolve until market clearing conditions are met
        if np.isnan(equilibrium_prices).any() or np.isinf(equilibrium_prices).any():
            return [1.0, 1.0]  # Return initial guesses if the solution is invalid
        return equilibrium_prices

    def calculate_market_conditions(self, p1, p2):
        """ Method created to calculate the market conditions for goods 1 and 2 given prices p1 and p2. 
            The method prints the results of the calculations to check if the markets clear or not.
            Parameters: 
                p1: price of good 1
                p2: price of good 2
        """
        self.set_prices(p1, p2)
        optimal_l = self.maximize_utility()  # Find the optimal labor supply
        
        # Calculate labor demand for each good
        l1_star = self.optimal_labor_demand(p1, self.w)
        l2_star = self.optimal_labor_demand(p2, self.w)
        
        # Calculate production for each good
        y1_star = self.optimal_production(l1_star)
        y2_star = self.optimal_production(l2_star)
        
        # Calculate consumption for each good
        pi_1 = self.optimal_profit(p1, self.w)
        pi_2 = self.optimal_profit(p2, self.w)
        c1_star, c2_star = self.consumer_behavior(optimal_l, pi_1, pi_2)
        
        # Ensure the outputs are scalars if they are numpy arrays
        y1_star = y1_star.item() if isinstance(y1_star, np.ndarray) and y1_star.size == 1 else y1_star
        c1_star = c1_star.item() if isinstance(c1_star, np.ndarray) and c1_star.size == 1 else c1_star
        y2_star = y2_star.item() if isinstance(y2_star, np.ndarray) and y2_star.size == 1 else y2_star
        c2_star = c2_star.item() if isinstance(c2_star, np.ndarray) and c2_star.size == 1 else c2_star

        # Print results
        print(f"Market Conditions at p1 = {p1:.2f}, p2 = {p2:.2f}:")
        print(f"  Production of Good 1: {y1_star:.2f}")
        print(f"  Consumption of Good 1: {c1_star:.2f}")
        print(f"  -> Market 1 {'clears' if np.isclose(y1_star, c1_star) else 'does not clear'}")
        
        print(f"  Production of Good 2: {y2_star:.2f}")
        print(f"  Consumption of Good 2: {c2_star:.2f}")
        print(f"  -> Market 2 {'clears' if np.isclose(y2_star, c2_star) else 'does not clear'}")

    def calculate_swf(self, l, kappa):
        """
        Calculates the Social Welfare Function (SWF) given the utility and optimal production of good 2.
        Parameters:
            l: Labor supply.
            kappa: Social cost of carbon emitted by production of y in equilibrium.
        Returns:
            The calculated SWF.
        """
        utility = self.utility_function(l)
        optimal_l2 = self.optimal_labor_demand(self.p2, self.w)
        y2_star = self.optimal_production(optimal_l2)
        swf = utility - kappa * y2_star
        return swf

    def calculate_swf_for_different_taus(self, tau_values, kappa):
        """
        Calculates the SWF for different values of tau.
        Parameters:
            tau_values: An array of tau values to iterate over.
            kappa: the social cost of carbon emitted by production of y in equilibrium
        Returns:
            A list of tuples containing tau values and their corresponding SWF values.
        """
        swf_values = []
        
        # Iterate over the tau values
        for tau in tau_values:
            self.tau = tau  # Set the new value of tau
            equilibrium_prices = self.market_clearing_prices()  # Recalculate prices for the current tau
            self.set_prices(*equilibrium_prices)  # Set the new prices for goods 1 and 2
            optimal_l = self.maximize_utility()  # Find the optimal labor supply given the new prices
            if optimal_l is not None:  # Check if the optimization was successful
                swf = self.calculate_swf(optimal_l, kappa)  # Calculate the SWF
                swf_values.append((tau, swf))  # Append the tau and SWF values to the list

        return swf_values

    def plot_swf_for_different_taus(self, tau_values, kappa):
        """
        Plots the SWF for different values of tau.
        """
        # Calculate the SWF values for different tau values
        swf_values = self.calculate_swf_for_different_taus(tau_values, kappa)

        taus, swfs = zip(*swf_values)

        plt.figure(figsize=(10, 6))
        plt.plot(taus, swfs, marker='o', linestyle='-', color='blue')
        plt.xlabel('Tax Rate ($\\tau$)')
        plt.ylabel('Social Welfare Function (SWF)')
        plt.title('SWF for Different Values of $\\tau$')
        plt.grid(True)
        plt.show()
    
    def calculate_swf_for_tau(self, tau, kappa):
        """
        Calculates the SWF for a given value of tau.
        Parameters:
            tau: The tax rate.
            kappa: The social cost of carbon emitted by production of y in equilibrium.
        Returns: 
            The SWF for the given tau.
        """
        self.tau = tau
        equilibrium_prices = self.market_clearing_prices()  # Recalculate prices for the current tau
        if np.isnan(equilibrium_prices).any() or np.isinf(equilibrium_prices).any():
            return np.inf  # Return a large value if prices are invalid

        self.set_prices(*equilibrium_prices)  # Set the new prices
        optimal_l = self.maximize_utility()
        if optimal_l is not None and optimal_l > 0 and not np.isnan(optimal_l) and not np.isinf(optimal_l):
            swf = self.calculate_swf(optimal_l, kappa)
            if not np.isnan(swf) and not np.isinf(swf):
                return swf
        return np.inf  # Return a large value if the optimization fails or SWF is invalid

    def find_optimal_tau(self, kappa, tau_bounds):
        """
        Finds the value of tau that gives the highest SWF using numerical optimization.
        Parameters:
            kappa: The social cost of carbon emitted by production of y in equilibrium.
            tau_bounds: The bounds for the tax rate tau.
        Returns: 
            The optimal value of tau and the corresponding SWF.
        """
        result = minimize_scalar(lambda tau: -self.calculate_swf_for_tau(tau, kappa), bounds=tau_bounds, method='bounded')
        optimal_tau = result.x
        optimal_swf = -result.fun
        return optimal_tau, optimal_swf

    def calculate_optimal_tau_and_market_conditions(self, kappa, tau_bounds):
        """
        Finds the optimal value of tau and calculates the equilibrium prices, consumptions, productions, and lump-sum transfer.
        Parameters:
            kappa: The social cost of carbon emitted by production of y in equilibrium.
            tau_bounds: The bounds for the tax rate tau.
        Returns:
            A dictionary containing the optimal tau, optimal SWF, equilibrium prices, consumptions, productions, and lump-sum transfer.
        """
        # Find the optimal tau and the corresponding SWF
        optimal_tau, optimal_swf = self.find_optimal_tau(kappa=kappa, tau_bounds=tau_bounds)

        # Set the tau to the optimal value
        self.tau = float(optimal_tau)  # Ensure it is a scalar float

        # Calculate the equilibrium prices
        equilibrium_prices = self.market_clearing_prices()
        p1, p2 = equilibrium_prices

        # Set the prices in the ProductionEconomy instance
        self.set_prices(p1, p2)

        # Calculate the optimal labor supply
        optimal_l = self.maximize_utility()

        # Calculate the production and consumption values
        optimal_l1 = self.optimal_labor_demand(p1, self.w)
        optimal_l2 = self.optimal_labor_demand(p2, self.w)
        y1 = self.optimal_production(optimal_l1)
        y2 = self.optimal_production(optimal_l2)
        pi_1 = self.optimal_profit(p1, self.w)
        pi_2 = self.optimal_profit(p2, self.w)
        c1, c2 = self.consumer_behavior(optimal_l, pi_1, pi_2)

        # Calculate the lump-sum transfer T
        T = self.T

        # Convert NumPy arrays to scalars if needed
        y1 = float(y1) if isinstance(y1, np.ndarray) else y1
        c1 = float(c1) if isinstance(c1, np.ndarray) else c1
        y2 = float(y2) if isinstance(y2, np.ndarray) else y2
        c2 = float(c2) if isinstance(c2, np.ndarray) else c2
        optimal_tau = float(optimal_tau) if isinstance(optimal_tau, np.ndarray) else optimal_tau
        T = float(T) if isinstance(T, np.ndarray) else T
        optimal_swf = float(optimal_swf) if isinstance(optimal_swf, np.ndarray) else optimal_swf

        # Return the results as a dictionary
        return {
            "optimal_tau": optimal_tau,
            "optimal_swf": optimal_swf,
            "equilibrium_prices": (p1, p2),
            "consumptions": (c1, c2),
            "productions": (y1, y2),
            "lump_sum_transfer": T
        }