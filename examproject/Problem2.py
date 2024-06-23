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
        Method to simulate the utility for each career choice

        args:
            j (int): career choice
            epsilon (np.array): random error term of utility
            sigma (float): standard deviation of error term
            v (np.array): utility for each career choice
        returns:
            expected_utilities: expected utility for each career choice
            realized_utilities: average realized utility for each career choice
        """
        np.random.seed(2024)  # For reproducibility
        expected_utilities = np.zeros(self.J) # Initialize the expected utility for each career choice
        realized_utilities = np.zeros(self.J) # Initialize the realized utility for each career choice
        
        for j in range(self.J):
            epsilon = np.random.normal(0, self.sigma, self.K) # Draw epsilon for each career choice j from a normal distribution K times
            expected_utility = self.v[j] + 0 # Calculate the expected utility for each career choice j. Since epsilon is drawn from a normal distribution with mean 0, the expected utility is equal to the utility of the career choice j
            realized_utility = self.v[j] + epsilon # Calculate the realized utility for each career choice j by adding the error term to the utility of the career choice j
            
            expected_utilities[j] = expected_utility # Store the expected utility for each career choice j
            realized_utilities[j] = np.mean(realized_utility) # Store the average realized utility for each career choice j
        
        return expected_utilities, realized_utilities
    
    # Method to get the results of the simulation
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
        career_choices = np.zeros((self.N, self.K)) # Initialize the career choice for each friend
        prior_expectations = np.zeros((self.N, self.K)) # Initialize the prior expected utility for each friend
        realized_utilities = np.zeros((self.N, self.K)) # Initialize the realized utility for each friend

        # Loop over the number of friends (N) and the number of simulations (K)
        for i in range(1, self.N + 1): # Loop over the number of friends
            for k in range(self.K): # K simulations for each friend
                epsilon_friends = np.random.normal(0, self.sigma, (self.J, i)) #1. Draw epsilon for each friend in each career track
                epsilon_own = np.random.normal(0, self.sigma, self.J) #1. Draw epsilon for own utility in each career track
                
                prior_exp_utility = self.v + np.mean(epsilon_friends, axis=1) #Step 1. Calculate prior expected utility of each career track
                chosen_career = np.argmax(prior_exp_utility) #Step 2. Choose career track with highest prior expected utility
                
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

    """
    Question 3: Switching Scenario
    """
    def simulate_switching_scenario(self):
        """
        Simulate the switching scenario to calculate the switching decisions and new realized utilities
        """
        np.random.seed(2024)  # For reproducibility
        career_choices, prior_expectations, realized_utilities = self.simulate_new_scenario() # Simulate the career choices, prior expected utilities, and realized utilities
        
        switch_decisions = np.zeros((self.N, self.K)) # Initialize the switching decisions
        new_realized_utilities = np.zeros((self.N, self.K)) # Initialize the new realized utilities
        new_subjective_utilities = np.zeros((self.N, self.K)) # Initialize the new subjective utilities

        for i in range(1, self.N + 1): # Loop over the number of friends
            for k in range(self.K): # Loop over the number of simulations
                chosen_career = int(career_choices[i - 1, k]) # Get the chosen career
                true_utility = realized_utilities[i - 1, k] # Get the true utility
                
                # Calculate new prior expected utilities including switching cost
                new_prior_exp_utility = np.zeros(self.J) # Initialize the new prior expected utilities
                for j in range(self.J): # Loop over the career choices
                    if j != chosen_career: # If the career choice is different from the chosen career
                        new_prior_exp_utility[j] = prior_expectations[i - 1, k] - self.c # Calculate the new prior expected utility with switching cost
                    else: # If the career choice is the same as the chosen career
                        new_prior_exp_utility[j] = true_utility # Set the new prior expected utility to the true utility
                
                new_chosen_career = np.argmax(new_prior_exp_utility) # Choose the career with the highest new prior expected utility
                new_subjective_utilities[i - 1, k] = new_prior_exp_utility[new_chosen_career] # Store the new subjective utility
                if new_chosen_career != chosen_career: # If the new chosen career is different from the chosen career
                    switch_decisions[i - 1, k] = 1 # Set the switching decision to 1
                    new_realized_utilities[i - 1, k] = self.v[new_chosen_career] + np.random.normal(0, self.sigma) - self.c # Calculate the new realized utility including the switching cost
                else: # If the new chosen career is the same as the chosen career
                    new_realized_utilities[i - 1, k] = true_utility # Set the new realized utility to the true utility
        
        return switch_decisions, new_realized_utilities, new_subjective_utilities, career_choices # Return the switching decisions, new realized utilities, new subjective utilities, and career choices

    def visualize_switching_results(self, switch_decisions, new_realized_utilities, new_subjective_utilities, career_choices):
        """Visualize the results of the switching scenario simulation"""
        avg_new_realized_utilities = np.mean(new_realized_utilities, axis=1) # Calculate the average new realized utilities
        avg_new_subjective_utilities = np.mean(new_subjective_utilities, axis=1) # Calculate the average new subjective utilities

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.N + 1), avg_new_subjective_utilities, label='Avg Subjective Expected Utility', linestyle='-')
        plt.xlabel('Number of Friends (Fi)')
        plt.ylabel('Utility')
        plt.title('Average Subjective Expected Utility After Switching')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.N + 1), avg_new_realized_utilities, label='Avg Realized Utility After Switching', linestyle='--')
        plt.xlabel('Number of Friends (Fi)')
        plt.ylabel('Realized Utility')
        plt.title('Average Realized Utility After Switching')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Calculate conditional switching rates
        for j in range(self.J):
            cond_switch_rates = np.zeros(self.N)
            for i in range(1, self.N + 1):
                initial_choice = career_choices[i - 1, :] == j
                if np.sum(initial_choice) > 0:
                    cond_switch_rates[i - 1] = np.mean(switch_decisions[i - 1, initial_choice])
            
            plt.plot(range(1, self.N + 1), cond_switch_rates, label=f'Initial Career {j + 1}')

        plt.xlabel('Number of Friends (Fi)')
        plt.ylabel('Conditional Switching Rate')
        plt.title('Conditional Switching Rate Based on Initial Career Choice')
        plt.legend()
        plt.show()