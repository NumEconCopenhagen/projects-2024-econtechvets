from sympy import symbols, Eq, solve, lambdify, simplify, latex
from types import SimpleNamespace
from IPython.display import display, Math, HTML
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

class IS_LM_model_analytical():
    
    def __init__(self, **kwargs):
        '''
        Initializes the model with default parameters
        kwargs allow any parameter in the par namespace to be overridden by the user
        '''
        self.par = par = SimpleNamespace()  # Create a namespace object for parameters
        self.sol = sol = SimpleNamespace()  # Create a namespace object for solution results
        self.sim = sim = SimpleNamespace()  # Create a namespace object for simulation results

        # Set default parameters
        self.setup()

        # Update parameters with user input
        for key, value in kwargs.items():
            setattr(par, key, value)

    def setup(self):
        '''
        Set default parameters
        '''
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
        '''
        Derive the IS-LM equations as a Python function using SymPy and lambdify
        '''
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
        '''
        Solve the IS-LM model using the Python functions derived from the symbolic equations
        '''
        par = self.par

        # Calculate the equilibrium values using the derived functions
        self.sol.Y = self.Y_func(par.a, par.b, par.T, par.c, par.d, par.G, par.M, par.P, par.e, par.f)
        self.sol.r = self.r_func(par.a, par.b, par.T, par.c, par.d, par.G, par.M, par.P, par.e, par.f)

    def print_IS_LM_equations(self):
        '''
        Print the IS-LM equations using display for better formatting
        '''
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
        '''
        Print the equilibrium solution
        '''
        sol = self.sol
        # Use HTML to make the text bold
        display(HTML('<strong>Equilibrium Solution (Analytical):</strong>'))
        print(f'Equilibrium Output (Y): {sol.Y:.2f}')
        print(f'Equilibrium Interest Rate (r): {sol.r:.2f}')

    def plot_IS_LM_curves(self):
        par = self.par

        # Calculate equilibrium values
        Y_eq = self.sol.Y
        r_eq = self.sol.r

        # Range of output (Y) values
        Y_values = np.linspace(0, Y_eq * 2, 100)  # Increase the range for better visibility

        # Calculate corresponding interest rates for the LM curve
        r_values = (par.e * Y_values - par.M / par.P) / par.f

        # Calculate corresponding interest rates for the IS curve
        IS_values = (1 / par.d) * (par.a + par.c - par.d * par.T + par.G - (1 - par.b) * Y_values)

        # Plot IS and LM curves
        plt.figure(figsize=(10, 6))
        plt.plot(Y_values, IS_values, label='IS curve', color='blue', linestyle='--')
        plt.plot(Y_values, r_values, label='LM curve', color='red')

        # Plot equilibrium point
        plt.scatter(Y_eq, r_eq, color='black', label='Equilibrium')

        plt.title('IS-LM Model')
        plt.xlabel('Output (Y)')
        plt.ylabel('Interest Rate (r)')
        plt.ylim(min(r_values.min(), IS_values.min()) - 5, max(r_values.max(), IS_values.max()) + 5)  # Adjust y-axis limit
        plt.grid(True)
        plt.legend()
        plt.show()


class IS_LM_numerical():
    def __init__(self, analytical_model=None, **kwargs):
        '''
        Initializes the numerical model
        As the analytical model is provided, this inherits its parameters
        Otherwise, kwargs would allow any parameter to be set
        '''
        self.par = par = SimpleNamespace()  # Create a namespace object for parameters
        self.sol = sol = SimpleNamespace()  # Create a namespace object for solution results

        if analytical_model:
            # Copy parameters from the analytical model
            for key, value in vars(analytical_model.par).items():
                setattr(par, key, value)
        else:
            # Set default parameters if no analytical model is provided
            self.setup()

            # Update parameters with user input
            for key, value in kwargs.items():
                setattr(par, key, value)

    def setup(self):
        '''
        Set default parameters
        '''
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
        '''
        Solve the IS-LM model numerically using SciPy's minimize
        '''
        par = self.par

        # Objective function to minimize
        def objective(vars):
            Y, r = vars
            IS = Y - (par.a - par.b * par.T + par.c - par.d * r + par.G) / (1 - par.b)
            LM = r - (par.e * Y - par.M / par.P) / par.f
            return IS**2 + LM**2  # Sum of squared errors (SSE)

        # Optimize the system using SciPy's minimize function
        result = minimize(objective, initial_guess, method='BFGS')

        # Extract the results
        sol_Y, sol_r = result.x

        # Store the results
        self.sol.Y = sol_Y
        self.sol.r = sol_r
        self.initial_guess = initial_guess

    def print_solution(self):
        '''
        Print the equilibrium solution
        '''
        sol = self.sol
        initial_guess = self.initial_guess
        # Use HTML to make the text bold
        display(HTML('<strong>Equilibrium Solution (Numerical):</strong>'))
        print(f'For initial guesses: Y = {initial_guess[0]:.1f} and r = {initial_guess[1]:.1f}: '
              f'the equilibrium output and interest rate are (Y, r) = ({sol.Y:.2f}, {sol.r:.2f})')