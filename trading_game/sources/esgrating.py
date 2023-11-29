import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


class ESGratings:
    def __init__(self) -> None:
        pass

    def get_esgdata(self, esgratings_df, info_df):
        esgratings_df = esgratings_df[['Symbol', 'Sector', 'Total ESG Risk score']]
        sector_medians = esgratings_df.groupby('Sector')['Total ESG Risk score'].median()

        # Create a DataFrame with the ESG scores
        full_esg_df = pd.DataFrame(index=info_df.index)
        full_esg_df['ESG Score'] = np.nan  # Initialize all scores as NaN

        # Iterate over all companies in info_df
        for company in full_esg_df.index:
            if company in esgratings_df['Symbol'].values:
                # Use the actual ESG score if available
                full_esg_df.at[company, 'ESG Score'] = esgratings_df[esgratings_df['Symbol'] == company]['Total ESG Risk score'].values[0]
            else:
                # Retrieve the sector from info_df
                sector = info_df.at[company, 'RBICS Economy']
                # Use the sector median if the sector is available
                if sector in sector_medians.index:
                    full_esg_df.at[company, 'ESG Score'] = sector_medians[sector]

        # Handling cases where the sector is unknown or no median is available
        full_esg_df['ESG Score'].fillna(full_esg_df['ESG Score'].median(), inplace=True)
        full_esg_df['Inverted_ESG'] = -full_esg_df['ESG Score']
        
        return full_esg_df
    
    def plot_ESGscore_candlechart(self, esgratings_df, info_df, threshold):
        merged_df = pd.merge(info_df, esgratings_df, on='Ticker')
        # Calculating statistics
        stats_df = merged_df.groupby('RBICS Economy')['ESG Score'].agg(['mean', 'min', 'max', 'std'])
        fig, ax = plt.subplots()

        # Bar plot for the mean
        stats_df['mean'].plot(kind='bar', ax=ax, color='skyblue', label='Average')

        # Scatter plot for min and max
        ax.scatter(stats_df.index, stats_df['min'], color='green', label='Min')
        ax.scatter(stats_df.index, stats_df['max'], color='red', label='Max')

        # Plot lines for standard deviation
        for sector in stats_df.index:
            mean = stats_df.at[sector, 'mean']
            std_dev = stats_df.at[sector, 'std']
            ax.plot([sector, sector], [mean - std_dev, mean + std_dev], color='black', marker='_', label='Std Dev' if sector == stats_df.index[0] else "")

        # Adding a horizontal line for the threshold
        ax.axhline(y=threshold, color='purple', linestyle='--', label='Threshold (25)')
        plt.title('ESG Score Statistics by Economy Sector')
        plt.xlabel('Economy Sector')
        plt.ylabel('ESG Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def filter_esgratings(self, esg_ratings, threshold = 25):
        filtered_esgcompanies = esg_ratings[esg_ratings['ESG Score'] <= threshold]
        print(f'After using threshold for esg ratings ({threshold}), we are still left with {len(filtered_esgcompanies)} companies')
        return filtered_esgcompanies.index
    
