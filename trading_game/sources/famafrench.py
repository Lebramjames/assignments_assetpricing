import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import statsmodels.api as sm


class FamaFrench:
    def __init__(self, rf_rate) -> None:
        self.rf_rate = rf_rate
        pass



    # %% Forecast Fama French, since the last updated date is 29th of september we need the rest for forecasting:
    def test_stationarity(self, timeseries):
        dftest = adfuller(timeseries, autolag = 'AIC')
        return dftest[1]

    def forecast_VAR_ff(self, factors_df, daily_logreturns, file_dir = r'data/', max_lags = 120, significance = 0.05):

        df = factors_df.drop('RF', axis = 1).copy()
        # last_date = df.index[-1]
        # required_last_date = daily_logreturns.index[-1]
        # new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=required_last_date, freq='B')

        forecast_start_date = pd.to_datetime('2023-01-01')
        forecast_end_date = pd.to_datetime('2023-12-31')
        new_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='B')


        n_forecast_steps = len(new_dates)

        for column in df.columns: 
            if self.test_stationarity(df[column]) > significance:
                df[column] = df[column].diff().dropna()

        model = VAR(df)
        lag_order = model.select_order(maxlags=max_lags)
        optimal_lag = lag_order.selected_orders['aic']
        fitted_model = model.fit(optimal_lag)

        # old
        # forecasted_values = fitted_model.forecast(df.values[-optimal_lag:], steps=n_forecast_steps)
        # forecasted_df = pd.DataFrame(forecasted_values, columns=df.columns)
        # forecasted_df.index = new_dates

        # Extract data from 2020 to "now" (end of 2022)
        historical_data = df.loc['2020-01-01':'2022-12-31']

        # Forecast Fama-French factors for 2023
        forecasted_values = fitted_model.forecast(historical_data.values[-optimal_lag:], steps=n_forecast_steps)
        forecasted_df = pd.DataFrame(forecasted_values, columns=df.columns)
        forecasted_df.index = new_dates

        forecasted_df.to_parquet(file_dir + 'ForecastedFF_VAR.par')

        return forecasted_df
    
    # %% Regression:    
    def regression_famafrench(self, excess_returns, factors_df, factor_names):
        num_factors = len(factor_names)
        df_params = pd.DataFrame(index=excess_returns.columns, columns=factor_names )
        df_signif = pd.DataFrame(index=excess_returns.columns, columns=factor_names)
        df_params['alpha'] = None
        df_signif['alpha'] = None

        for stock in excess_returns.columns:
            y = excess_returns[stock]
            X = sm.add_constant(factors_df[factor_names])
            model = sm.OLS(y, X)
            results = model.fit()
            df_params.loc[stock, factor_names] = results.params.values[1:]  # Exclude the constant term
            df_params.loc[stock, 'alpha'] = results.params[0]  # Include the constant term

            df_signif.loc[stock, factor_names] = results.tvalues[1:]  # Exclude the constant term
            df_signif.loc[stock, 'alpha'] = results.tvalues[0]  # Include the constant term
            # print(df_signif)

        return df_params, df_signif


    def run_famafrench(self, factors_df, df_params, df_signif, factor_names):
        avg_ret = factors_df[factor_names].mean(axis=0)


        # Set parameter values that are not significant to zero
        df_params[df_signif.abs() < 1.645] = 0

        # Calculate returns per stock by multiplying parameter values with corresponding factor returns
        df_params['avg_ret'] = df_params[factor_names].dot(avg_ret) + df_params['alpha']
        df_params['avg_ret'] = df_params['avg_ret'] + self.rf_rate / 252

        return df_params


    def select_topcompanies_avgreturn(self, df_params, n_largest=200):
        df_params['avg_ret'] = pd.to_numeric(df_params['avg_ret'], errors='coerce')
        top_indices = df_params.nlargest(n_largest, 'avg_ret').index

        return top_indices
    

    # %% IML:
    def calc_daily_illq(self, returns_daily_df, dollar_vols_daily_df):
        # Calculate daily illiquidity values
        daily_volumes = dollar_vols_daily_df
        df_illiquidity = 10000000 * returns_daily_df.abs() / daily_volumes
        
        return df_illiquidity

    def calculate_daily_quintile_returns(self, df, quantiles, quintile):
        mask = quantiles.shift(1) == quintile  # Shift by one day to use the previous day's quintile
        # Equally weighted average of returns
        return df[mask].mean(axis=1)
    

    def calc_monthly_illq(self, returns_monthly_df, dollar_vols_daily_df):
        # to ensure taking the equally weighted average of the daily I LLIQ values in a given month we first ahve to sum each month
        monthly_volumes = dollar_vols_daily_df.resample('M').sum()
        # since the volumes are set to 31th of each month, we have to of set the month to 01 
        monthly_volumes.index = monthly_volumes.index.to_period('M').to_timestamp('M')
        monthly_volumes.index = monthly_volumes.index - pd.DateOffset(months=1)
        # aligning the monthly volumn with the monthly return
        monthly_volumes = monthly_volumes.reindex(returns_monthly_df.index, method='ffill')

        df_illiquidity = 1000000 *returns_monthly_df.abs() / monthly_volumes
        
        return df_illiquidity
    
    def calculate_monthly_quintile_returns(self, df, quantiles, quintile):
        mask = quantiles.shift(1) == quintile  # Shift by one month to use previous month's quintile
        # Equally weighted average of returns
        return df[mask].mean(axis=1)

    def calc_iml_factor(self, returns, volume, nr_quant, freq='daily'):
        if freq == 'daily':
            # daily
            df_illiquidity = self.calc_daily_illq(returns, volume).dropna()
            quintiles = df_illiquidity.apply(lambda x: pd.qcut(x, nr_quant, labels=False, duplicates='drop'), axis=1)
            quintiles = quintiles.applymap(lambda x: 4 - x if not pd.isnull(x) else x)

            illiquid_returns = self.calculate_daily_quintile_returns(returns, quintiles, 0)  # most illiquid quintile
            liquid_returns = self.calculate_daily_quintile_returns(returns, quintiles, 4)  # most liquid quintile
        if freq == 'monthly':
            # monthly
            df_illiquidity = self.calc_monthly_illq(returns, volume)
            quintiles = df_illiquidity.apply(lambda x: pd.qcut(x, nr_quant, labels=False, duplicates='drop'), axis=1)

            # Inverts the ranking of the quantiles (with the most illiquid stocks in quintile one and the most liquid stocks in quintile five.)
            quintiles = quintiles.applymap(lambda x: 4 - x if not pd.isnull(x) else x)

            illiquid_returns = self.calculate_monthly_quintile_returns(returns, quintiles, 0)  # most illiquid quintile
            liquid_returns = self.calculate_monthly_quintile_returns(returns, quintiles, 4)  # most liquid quintile

        iml_factor = illiquid_returns - liquid_returns
        return iml_factor
