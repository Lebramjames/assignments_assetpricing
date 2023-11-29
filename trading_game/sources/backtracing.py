import numpy as np


class Backtracing: 
        def __init__(self) -> None:
                pass

        def calculate_forecast_error(self, actual, forecast):
            error = actual - forecast
            mae = np.mean(np.abs(error))
            rmse = np.sqrt(np.mean(error**2))
            mape = np.mean(np.abs(error / actual)) * 100
            return mae, rmse, mape