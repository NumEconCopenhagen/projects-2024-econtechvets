import numpy as np
import matplotlib.pyplot as plt

class CareerChoiceClass:
    """ Class for simulating career choice model"""
    def __init__(self):
        """ Initialize parameters using the given values"""
        self.J = 3
        self.N = 10
        self.K = 10000
        self.F = np.arange(1, self.N+1)
        self.sigma = 2
        self.v = np.array([1, 2, 3])
        self.c = 1
    
    """
    Question 1:
    """
    def simulate_utilities(self):
        """
        args:
            j: (int) career choice
            epsilon (np.array) random error term of utility
            sigma: (float) standard deviation of error ter
            v: (np.array) utility for each career choice
        returns:
            expected_utilities: np.array: expected utility for each career choice
            realized_utilities: np.array: average realized utility for each career choice
        """
        np.random.seed(2024)  # For reproducibility
        expected_utilities = np.zeros(self.J)
        realized_utilities = np.zeros(self.J)
        

        for j in range(self.J):
            epsilon = np.random.normal(0, self.sigma, self.K)
            expected_utility = self.v[j] # Use the base utility directly as the theoretical expected value of epsilon is 0
            realized_utility = self.v[j] + epsilon
            
            expected_utilities[j] = expected_utility
            realized_utilities[j] = np.mean(realized_utility)
        
        return expected_utilities, realized_utilities
    
    def get_results(self):
        expected_utilities, realized_utilities = self.simulate_utilities()
        return expected_utilities, realized_utilities

    """
    Question 2: Friends
    """
    def simulate_new_scenario(self):
        """
        args:
            N: (int) number of friends
            J: (int) number of career choices
            K: (int) number of simulations
            sigma: (float) standard deviation of error term
            v: (np.array) utility for each career choice
        returns:
            career_choices: (np.array) career choice for each friend
            prior_expectations: (np.array) prior expected utility for each friend
            realized_utilities: (np.array) realized utility for each friend
        """
        np.random.seed(2024)  # For reproducibility
        career_choices = np.zeros((self.N, self.K))
        prior_expectations = np.zeros((self.N, self.K))
        realized_utilities = np.zeros((self.N, self.K))

        for i in range(1, self.N + 1):
            for k in range(self.K):
                epsilon_friends = np.random.normal(0, self.sigma, (self.J, i)) #1. Draw epsilon for each friend
                epsilon_own = np.random.normal(0, self.sigma, self.J) #1. Draw epsilon for own utility
                
                prior_exp_utility = self.v + np.mean(epsilon_friends, axis=1) #1. Calculate prior expected utility of each career track
                chosen_career = np.argmax(prior_exp_utility) # 2. Choose career with "highest prior expected utility"
                
                career_choices[i - 1, k] = chosen_career #3. store career choice
                prior_expectations[i - 1, k] = prior_exp_utility[chosen_career] #3. store prior expected utility
                realized_utilities[i - 1, k] = self.v[chosen_career] + epsilon_own[chosen_career] #3. store realized utility

        return career_choices, prior_expectations, realized_utilities

    def visualize_results(self, career_choices, prior_expectations, realized_utilities):
        avg_prior_expectations = np.mean(prior_expectations, axis=1)
        avg_realized_utilities = np.mean(realized_utilities, axis=1)

        plt.figure(figsize=(12, 6))

        for j in range(self.J):
            shares = np.mean(career_choices == j, axis=1)
            plt.plot(range(1, self.N + 1), shares, label=f'Career {j + 1}')

        plt.xlabel('Number of Friends (Fi)')
        plt.ylabel('Share of Graduates Choosing Career')
        plt.legend()
        plt.title('Share of Graduates Choosing Each Career Based on Number of Friends')
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, self.N + 1), avg_prior_expectations, label='Avg Prior Expected Utility')
        plt.plot(range(1, self.N + 1), avg_realized_utilities, label='Avg Realized Utility', linestyle='--')
        plt.xlabel('Number of Friends (Fi)')
        plt.ylabel('Utility')
        plt.legend()
        plt.title('Average Subjective Expected Utility and Realized Utility')
        plt.show()
