import mlflow
import pandas as pd
from prefect import flow
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from create_adj_matrices import load_taxi_from_api, load_zone_lookup_from_api, get_manhattan_data

mlflow_tracking_uri = 'http://localhost:5001'
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("OD-demand-models")

@flow
def historical_avg_demand():
    taxi_2024 = load_taxi_from_api()
    zone_lookup = load_zone_lookup_from_api()
    manhattan_df = get_manhattan_data(taxi_2024, zone_lookup)
    manhattan_df = manhattan_df.sort_values(by='tpep_pickup_datetime')
    manhattan_df['tpep_pickup_datetime'] = pd.to_datetime(manhattan_df['tpep_pickup_datetime'])

    manhattan_df.set_index('tpep_pickup_datetime', inplace=True)

    hourly_data = manhattan_df.groupby([pd.Grouper(freq='h'), 'PULocationID']).size().rename('demand').reset_index()

    hourly_data['hour'] = hourly_data['tpep_pickup_datetime'].dt.hour
    hourly_data['day_of_week'] = hourly_data['tpep_pickup_datetime'].dt.dayofweek

    historical_avg = hourly_data.groupby(['PULocationID', 'day_of_week', 'hour'])['demand'].mean().rename('historical_avg').reset_index()

    hourly_data = hourly_data.merge(historical_avg, on=['PULocationID', 'day_of_week', 'hour'], how='left')

    with mlflow.start_run():
        mape = mean_absolute_percentage_error(hourly_data['demand'], hourly_data['historical_avg'])
        mae = mean_absolute_error(hourly_data['demand'], hourly_data['historical_avg'])
        rmse = mean_squared_error(hourly_data['demand'], hourly_data['historical_avg'], squared=False)
        print(f'HA RMSE: {rmse:.4f} \nHA MAE: {mae:.4f} \nHA MAPE: {mape:.4f}')
        mlflow.log_metric('historical_avg_rmse', rmse)
        mlflow.log_metric('historical_avg_mae', mae)
        mlflow.log_metric('historical_avg_mape', mape)

if __name__ == '__main__':
    historical_avg_demand()