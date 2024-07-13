import torch
from torch.utils.data import DataLoader

import io
import requests
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from prefect import task, flow

from conv_lstm import Seq2Seq
from train_conv_lstm import AdjacencyMatrixDataset
from create_adj_matrices import load_zone_lookup_from_api, get_manhattan_data, create_adj_matrices

@task(log_prints=True)
def create_adj_matrices_pred(year='2023', month='01'):
    """Use 1 month of data for predictions"""
    print("Downloading data...")
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
    response = requests.get(url)
    content = io.BytesIO(response.content)
    table = pq.read_table(content)
    df = table.to_pandas()
    print("Data downloaded.")
    
    print("Loading zone lookup...")
    zone_lookup = load_zone_lookup_from_api()
    print("Zone lookup loaded.")
    
    print("Filtering Manhattan data...")
    df_manhattan = get_manhattan_data(df, zone_lookup)
    print("Manhattan data filtered.")
    
    print("Creating adjacency matrices...")
    adj_matrices = create_adj_matrices(df_manhattan, '2023-01-01', '2023-02-01')
    print("Adjacency matrices created.")
    
    return adj_matrices

@task(log_prints=True)
def load_model():
    print("Loading model...")
    device = 'mps' 
    params = {
                "input_dim": 1,
                "hidden_dim": 64,
                "kernel_size": (3, 3),
                "num_layers": 5,
                "batch_first": True
            }
    model = Seq2Seq(**params).to(device)
    model.load_state_dict(torch.load('model_2024-07-13 13:30:01.pth'))
    model.eval()
    print("Model loaded.")
    return model

@flow(log_prints=True)
def predict():
    adj_matrices = create_adj_matrices_pred()
    model = load_model()
    dataset = AdjacencyMatrixDataset(adj_matrices)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    with torch.no_grad():
        for i, sequence in enumerate(loader):
            print(f"Processing batch {i + 1}")
            inputs = sequence[:, :-1, :, :, :].to('mps')
            targets = sequence[:, 1:, :, :, :]  

            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = torch.stack(outputs)

            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            sample_idx = 0  # Choose a sample index from the batch
            output_sample = outputs_np[sample_idx]
            target_sample = targets_np[sample_idx]

            # Calculate the difference between predicted and ground truth frames
            diff = output_sample[0, 0] - target_sample[0, 0]

            num_columns = 3

            fig, axes = plt.subplots(1, num_columns, figsize=(15, 7))

            # Plot predicted output frame
            axes[0].imshow(output_sample[0, 0], cmap='viridis')  
            axes[0].set_title(f'Predicted')
            axes[0].axis('off') 

            # Plot ground truth for predicted output frame
            axes[1].imshow(target_sample[0, 0], cmap='viridis')  
            axes[1].set_title(f'GT')
            axes[1].axis('off') 
        
            # Plot difference between predicted and ground truth frames without normalization
            diff_rmse = mean_squared_error(output_sample[0, 0], target_sample[0, 0], squared=False)
            diff_plot = axes[2].imshow(diff, cmap='bwr')  
            axes[2].set_title(f'RMSE: {diff_rmse:.4f}')
            axes[2].axis('off') 

            # Add colorbar to the difference plot to show scale
            cbar = fig.colorbar(diff_plot, ax=axes[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(f'sample_conv_lstm_pred_img/output_{i}.png')
            plt.close()
            
            if i == 9:
                break

if __name__ == '__main__':
    predict()
