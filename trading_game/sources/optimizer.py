import numpy as np
import scipy.optimize as opt
import pandas as pd


class Optimizer:
    def __init__(self):
        pass

    def objective_function(self, weights: list, returns):

        mean_returns = np.mean(returns, axis=0)
        portfolio_return = weights @  mean_returns

        portfolio_std = np.sqrt(weights @ np.cov(returns.T) @ weights.T)
        # Calculate the correlation matrix from the covariance matrix
        # correlation_matrix = np.corrcoef(returns.T)

        # # Calculate the weighted average correlation
        # n = len(weights)
        # weighted_corr = sum(weights[i] * weights[j] * correlation_matrix[i, j]
        #                     for i in range(n) for j in range(n)) / n

        # # Define the correlation penalty
        # correlation_penalty = weighted_corr

        # Minimize the negative of the objective
        return -1 * (portfolio_return - 0.25 * portfolio_std)

    def calc_normalized_weights(self, optimal_weights, stock_names, threshold=0.005):

        # Filter values based on the threshold value (standard is 0.005)
        thresholded_weights = np.where(
            optimal_weights >= threshold, optimal_weights, 0)

        if np.sum(thresholded_weights) > 0:  # Prevent division by zero
            normalized_weights = thresholded_weights / \
                np.sum(thresholded_weights)
        else:
            # In case all are zero, which should not happen
            normalized_weights = thresholded_weights

        # Create a DataFrame from the stock names and their corresponding weights
        df_stocks_with_weights = pd.DataFrame({
            'Stock': stock_names,
            'Weight': normalized_weights
        })

        df_stocks_with_weights = df_stocks_with_weights[df_stocks_with_weights['Weight'] >= threshold]

        return df_stocks_with_weights

    def run_optimizer(self, n_stocks, returns):

        initial_weights = np.array([1 / n_stocks] * n_stocks)

        constraints = (
            {'type': 'ineq', 'fun': lambda weights: 0.85 -
                np.sum(weights)},  # Sum of weights >= 0.85
            {'type': 'ineq', 'fun': lambda weights: np.sum(
                weights) - 1.0}  # Sum of weights <= 1
        )
        bounds = tuple((0, 1) for x in range(n_stocks))
        optimized = opt.minimize(self.objective_function, initial_weights, args=(
            returns), bounds=bounds, constraints=constraints)  # Adjust the method as needed
        optimal_weights = optimized.x
        return optimal_weights
