from sympy import symbols, Eq, solve, lambdify, simplify, latex
from types import SimpleNamespace
from IPython.display import display, Math, HTML
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

class IS_LM_model_analytical():
    """
    A class representing the analytical approach to solving the IS-LM model.
    This class allows for symbolic derivation and analytical solving of the
    IS curve and LM curve equations in the context of macroeconomic analysis.
    """
    def __init__(self, **kwargs):
        """
        Initializes the IS-LM model with optional custom parameters.
        
        Parameters:
            **kwargs (dict): Arbitrary keyword arguments that are used to
                             override default model parameters.
        """
        self.par = par = SimpleNamespace()  # Create a namespace object for parameters
        self.sol = sol = SimpleNamespace()  # Create a namespace object for solution results
        self.sim = sim = SimpleNamespace()  # Create a namespace object for simulation results

        # Set default parameters
        self.setup()

        # Update parameters with user input
        for key, value in kwargs.items():
            setattr(par, key, value)

    def setup(self):
        """
        Sets the default parameters for the IS-LM model.
        """
        par = self.par

        # Model parameters (Updated)
        par.T = 100       # Taxes
        par.G = 400       # Government spending
        par.M = 1200      # Money supply
        par.P = 1         # Price level
        par.a = 200       # Autonomous consumption
        par.b = 0.7       # Marginal propensity to consume
        par.c = 300       # Autonomous investment
        par.d = 80        # Interest rate sensitivity of investment
        par.e = 0.6       # Sensitivity of money demand to changes in income
        par.f = 20        # Sensitivity of money demand to changes in interest rates

    def derive_IS_LM_equations(self):
        """
        Derives the IS and LM equations symbolically and prepares functions
        for calculating equilibrium values analytically.
        """
        # Define symbols
        Y, r = symbols('Y r')
        a, b, T, c, d, G, M, P, e, f = symbols('a b T c d G M P e f')

        # Define IS equation
        IS = Eq(Y, (a - b * T + c - d * r + G) / (1 - b))

        # Define LM equation
        LM = Eq(r, (e * Y - M / P) / f)

        # Solve the system of equations
        solution = solve((IS, LM), (Y, r))

        # Simplify the solution for clarity
        solution_Y = simplify(solution[Y])
        solution_r = simplify(solution[r])

        # Store symbolic solutions for printing
        self.symbolic_Y = solution_Y
        self.symbolic_r = solution_r

        # Convert symbolic solution to Python functions
        self.Y_func = lambdify((a, b, T, c, d, G, M, P, e, f), solution_Y, modules='numpy')
        self.r_func = lambdify((a, b, T, c, d, G, M, P, e, f), solution_r, modules='numpy')

        # Store the equations for display
        self.IS_eq = IS
        self.LM_eq = LM

    def solve_IS_LM_analytically(self):
        """
        Solves the IS-LM model analytically using the previously derived functions
        and updates the solution namespace with the equilibrium values of output (Y)
        and interest rate (r).
        """
        par = self.par

        # Calculate the equilibrium values using the derived functions
        self.sol.Y = self.Y_func(par.a, par.b, par.T, par.c, par.d, par.G, par.M, par.P, par.e, par.f)
        self.sol.r = self.r_func(par.a, par.b, par.T, par.c, par.d, par.G, par.M, par.P, par.e, par.f)

    def print_IS_LM_equations(self):
        """
        Displays the IS and LM equations in a formatted manner using LaTeX
        representation in Jupyter Notebooks.
        """
        IS_eq = self.IS_eq
        LM_eq = self.LM_eq

        display(HTML('<strong>The IS-LM equations can be rewritten into:</strong>'))
        
        # Append 'IS' and 'LM' on the right side of the equations
        display(Math(f'{latex(IS_eq)}\\quad \\text{{(IS)}}'))
        display(Math(f'{latex(LM_eq)}\\quad \\text{{(LM)}}'))

        display(HTML('<strong>The solution to Y and r are:</strong>'))
        display(Math(f'{latex(Eq(symbols("Y"), self.symbolic_Y))}'))
        display(Math(f'{latex(Eq(symbols("r"), self.symbolic_r))}'))

    def print_solution(self):
        """
        Prints the analytical solution for the equilibrium values of output (Y)
        and interest rate (r) in the IS-LM model.
        """
        sol = self.sol
        # Use HTML to make the text bold
        display(HTML('<strong>Equilibrium Solution (Analytical):</strong>'))
        print(f'Equilibrium Output (Y): {sol.Y:.2f}')
        print(f'Equilibrium Interest Rate (r): {sol.r:.2f}')

class IS_LM_numerical():
    """
    A class for numerical analysis of the IS-LM model which allows for dynamic
    adjustment of parameters and the use of numerical optimization to find
    equilibrium points. It supports custom tax functions and can inherit parameters
    from an analytical model.

    Attributes:
        par (SimpleNamespace): A namespace for storing model parameters.
        sol (SimpleNamespace): A namespace for storing solution results.
        tax_function (function): A custom function to calculate tax based on output.
    """
    def __init__(self, analytical_model=None, tax_function=None, **kwargs):
        """
        Initializes the numerical IS-LM model with parameters that can be inherited
        from an analytical model or specified directly.

        Parameters:
            analytical_model (IS_LM_model_analytical, optional): An instance of an analytical
                IS-LM model whose parameters are to be copied. If not provided, default
                parameters will be set.
            tax_function (callable, optional): A function that calculates tax based on
                output (Y). If not provided, a default tax function is used that returns a
                constant tax.
            **kwargs: Additional model parameters that can override the defaults or the
                inherited values from an analytical model.
        """
        self.par = SimpleNamespace()  # Create a namespace object for parameters
        self.sol = SimpleNamespace()  # Create a namespace object for solution results

        # Set the tax function
        self.tax_function = tax_function if tax_function is not None else lambda Y: 100  # Default tax as a constant

        if analytical_model:
            # Copy parameters from the analytical model
            for key, value in vars(analytical_model.par).items():
                setattr(self.par, key, value)
        else:
            # Set default parameters if no analytical model is provided
            self.setup()

            # Update parameters with user input
            for key, value in kwargs.items():
                setattr(self.par, key, value)

    def setup(self):
        """
        Sets default parameters for the IS-LM model if not provided during initialization.
        """
        par = self.par

        # Model parameters (Updated)
        par.T = 100       # Taxes
        par.G = 400       # Government spending
        par.M = 1200      # Money supply
        par.P = 1         # Price level
        par.a = 200       # Autonomous consumption
        par.b = 0.7       # Marginal propensity to consume
        par.c = 300       # Autonomous investment
        par.d = 80        # Interest rate sensitivity of investment
        par.e = 0.6       # Sensitivity of money demand to changes in income
        par.f = 20        # Sensitivity of money demand to changes in interest rates

    def solve_IS_LM_numerically(self, initial_guess=[1000, 5]):
        """
        Solves the IS-LM model numerically using the SciPy's minimize function to find
        the equilibrium values of output (Y) and interest rate (r).

        Parameters:
            initial_guess (list): A list of initial guesses [Y_initial, r_initial] for the
                                  numerical optimization algorithm.
        """
        par = self.par

        # Objective function to minimize
        def objective(vars):
            Y, r = vars
            T = self.tax_function(Y)  # Calculate T using the function
            IS = Y - (par.a - par.b * T + par.c - par.d * r + par.G) / (1 - par.b)
            LM = r - (par.e * Y - par.M / par.P) / par.f
            return IS**2 + LM**2

        # Optimize the system using SciPy's minimize function
        result = minimize(objective, initial_guess, method='BFGS')

        # Extract the results
        sol_Y, sol_r = result.x

        # Store the results
        self.sol.Y = sol_Y
        self.sol.r = sol_r
        self.initial_guess = initial_guess

    def print_solution(self):
        """
        Prints the calculated equilibrium values of output and interest rate using a
        visually formatted output incl. the initial guesses used for the numerical optimization.
        """
        sol = self.sol
        initial_guess = self.initial_guess
        # Use HTML to make the text bold
        display(HTML('<strong>Equilibrium Solution (Numerical):</strong>'))
        print(f'For initial guesses: Y = {initial_guess[0]:.1f} and r = {initial_guess[1]:.1f}: '
              f'the equilibrium output and interest rate are (Y, r) = ({sol.Y:.2f}, {sol.r:.2f})')
    
    def print_solution_2(self):
        """
        Prints the calculated equilibrium values of output and interest rate using a
        visually formatted output excl. the initial guesses used for the numerical optimization.
        """
        sol = self.sol
        initial_guess = self.initial_guess
        # Use HTML to make the text bold
        display(HTML('<strong>Equilibrium Solution (Numerical):</strong>'))
        print(f'the equilibrium output and interest rate are (Y, r) = ({sol.Y:.2f}, {sol.r:.2f})')
    
    def plot_curves(self, Y_range, r_range, *params):
        """
        Plots the numerical IS and LM curves for a given range of output (Y) and interest rate (r)
        """
        Y_values = np.linspace(Y_range[0], Y_range[1], 100)
        plt.figure(figsize=(10, 7))

        # Loop through each parameter set with an index
        for index, param_set in enumerate(params):
            # Update parameters
            for key, value in param_set.items():
                setattr(self.par, key, value)

            # Define line style, solid for the first set, dotted for others
            line_style = '-' if index == 0 else ':'

            # Recalculate IS and LM curves using updated parameters
            IS_curve = [(self.par.a + self.par.c + self.par.G - (1 - self.par.b) * Y - self.par.b * self.tax_function(Y)) / self.par.d for Y in Y_values]
            LM_curve = [(self.par.e * Y - self.par.M / self.par.P) / self.par.f for Y in Y_values]

            # Solve numerically for the intersection after updating parameters
            self.solve_IS_LM_numerically()
            sol_Y = self.sol.Y
            sol_r = self.sol.r

            plt.plot(Y_values, IS_curve, color='red', linestyle=line_style, label=f'IS (Params: {param_set})')
            plt.plot(Y_values, LM_curve, color='blue', linestyle=line_style, label=f'LM (Params: {param_set})')

            # Plot the intersection point
            plt.scatter(sol_Y, sol_r, color='black', s=100, zorder=5)

        plt.title('IS-LM Curves')
        plt.xlabel('Output (Y)')
        plt.ylabel('Interest Rate (r)')
        plt.xlim(Y_range[0], Y_range[1])
        plt.ylim(r_range[0], r_range[1])
        plt.legend()
        plt.grid(True)
        plt.show()