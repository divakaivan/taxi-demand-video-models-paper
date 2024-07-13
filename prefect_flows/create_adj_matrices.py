import io
import requests
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timedelta

from prefect import flow, task

@task(log_prints=True)
def load_taxi_from_api(year=2024, months=[1, 2, 3, 4]):
    monthly_dfs = [] 
    
    for i in months:
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{i:02d}.parquet'
        print(f'Getting data for Month {i:02d}')
        response = requests.get(url)

        content = io.BytesIO(response.content)
        table = pq.read_table(content)
        df = table.to_pandas()
        monthly_dfs.append(df)
    
    taxi_2024 = pd.concat(monthly_dfs, ignore_index=True)
    print('Concatenated data shape:', taxi_2024.shape)
    print('Done!')
    return taxi_2024

@task(log_prints=True)
def load_zone_lookup_from_api():
    print('Loading zone lookup...')
    url = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv'
    response = requests.get(url)
    zone_lookup = pd.read_csv(io.StringIO(response.text), sep=',')
    print('Zone lookup shape:', zone_lookup.shape)
    return zone_lookup

@task(log_prints=True)
def get_manhattan_data(taxi_2024, zone_lookup):
    print('Filtering to get only Manhattan data...')
    manhattan_ids = zone_lookup[zone_lookup['Borough'] == 'Manhattan']['LocationID'].values
    df_manhattan = taxi_2024[
        (taxi_2024['PULocationID'].isin(manhattan_ids)) &
        (taxi_2024['DOLocationID'].isin(manhattan_ids)) 
    ]

    cols_to_keep = ['tpep_pickup_datetime', 
                    'tpep_dropoff_datetime', 
                    'trip_distance', 
                    'PULocationID', 
                    'DOLocationID', 
                    'total_amount'
                    ]
    df_manhattan = df_manhattan[cols_to_keep]
    df_manhattan = df_manhattan.sort_values(by='tpep_pickup_datetime')
    print('Filtered Manhattan data shape:', df_manhattan.shape)
    return df_manhattan

@task(log_prints=True)
def create_adj_matrices(df_manhattan, start, end):
    print(f'Creating adjacency matrices from {start} to {end}...')
    df_manhattan["starttime"] = pd.to_datetime(df_manhattan["tpep_pickup_datetime"], format="%Y-%m-%d %H:%M:%S")
    df_manhattan["stoptime"] = pd.to_datetime(df_manhattan["tpep_dropoff_datetime"], format="%Y-%m-%d %H:%M:%S")

    start_date = datetime.strptime(f"{start} 00:00:01", "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(f"{end} 23:59:59", "%Y-%m-%d %H:%M:%S")
    interval = timedelta(minutes=60)
    bucket_elements = []
    ts_dfs = []
    while start_date <= end_date:
        
        bucket_elements.append(df_manhattan[((start_date + interval) >= df_manhattan["stoptime"])
                                    & (start_date <= df_manhattan["stoptime"])].shape[0])
        ts_dfs.append(df_manhattan[((start_date + interval) >= df_manhattan["stoptime"])
                                    & (start_date <= df_manhattan["stoptime"])])
        start_date += interval

    pickup_ids = df_manhattan['PULocationID'].unique()
    dropoff_ids = df_manhattan['DOLocationID'].unique()

    all_locations = pd.unique(pd.concat([pd.Series(pickup_ids), pd.Series(dropoff_ids)]))

    location_to_index = {location: index for index, location in enumerate(all_locations)}

    num_locations = len(all_locations)
    adj_matrices = []

    for ts_df in ts_dfs:
        adj_matrix = np.zeros((num_locations, num_locations), dtype=int)
        for _, row in ts_df.iterrows():
            pickup_index = location_to_index[row['PULocationID']]
            dropoff_index = location_to_index[row['DOLocationID']]
            adj_matrix[pickup_index][dropoff_index] += 1
        adj_matrices.append(adj_matrix)

    print('# of adj. matrices: ', len(adj_matrices), '; Each is of type: ', type(adj_matrices[0]))

    return adj_matrices

@flow(log_prints=True)
def create_adj_matrices_flow():
    taxi_2024 = load_taxi_from_api()
    zone_lookup = load_zone_lookup_from_api()
    df_manhattan = get_manhattan_data(taxi_2024, zone_lookup)
    adj_matrices = create_adj_matrices(df_manhattan, '2024-01-01', '2024-04-30')

    file_name = 'adj_matrices.npz'
    np.savez(file_name, *adj_matrices)
    print(f'Adjacency matrices saved to {file_name}')
    print('Done!')

if __name__ == '__main__':
    create_adj_matrices_flow()