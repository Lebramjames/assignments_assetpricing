import matplotlib.pyplot as plt

class Plotter:
    def __init__(self) -> None:
        pass

    def errorPlot(self, mae_VAR, rmse_VAR, mape_VAR, mae_avg, rmse_avg, mape_avg):
        # Fama-French factors
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Momentum']

        # MAE, RMSE, and MAPE for VAR and AVG forecasts
        mae_var = [mae_VAR[factor] for factor in factors]
        rmse_var = [rmse_VAR[factor] for factor in factors]
        mape_var = [mape_VAR[factor] for factor in factors]

        mae_avg = [mae_avg[factor] for factor in factors]
        rmse_avg = [rmse_avg[factor] for factor in factors]
        mape_avg = [mape_avg[factor] for factor in factors]

        # Set the width of the bars
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        index = range(len(factors))

        # Create subplots for MAE, RMSE, and MAPE
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Bar chart for MAE
        axs[0].bar(index, mae_var, bar_width, label='VAR')
        axs[0].bar([i + bar_width for i in index], mae_avg, bar_width, label='AVG')
        axs[0].set_xlabel('Fama-French Factors')
        axs[0].set_ylabel('MAE')
        axs[0].set_xticks([i + bar_width / 2 for i in index])
        axs[0].set_xticklabels(factors)
        axs[0].legend()

        # Bar chart for RMSE
        axs[1].bar(index, rmse_var, bar_width, label='VAR')
        axs[1].bar([i + bar_width for i in index], rmse_avg, bar_width, label='AVG')
        axs[1].set_xlabel('Fama-French Factors')
        axs[1].set_ylabel('RMSE')
        axs[1].set_xticks([i + bar_width / 2 for i in index])
        axs[1].set_xticklabels(factors)
        axs[1].legend()

        # Bar chart for MAPE
        axs[2].bar(index, mape_var, bar_width, label='VAR')
        axs[2].bar([i + bar_width for i in index], mape_avg, bar_width, label='AVG')
        axs[2].set_xlabel('Fama-French Factors')
        axs[2].set_ylabel('MAPE (%)')
        axs[2].set_xticks([i + bar_width / 2 for i in index])
        axs[2].set_xticklabels(factors)
        axs[2].legend()

        # Add a title
        plt.suptitle('Forecast Error Comparison: VAR vs. AVG')

        # Show the plots
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('ForecastErrorComparison.png')
        plt.show()