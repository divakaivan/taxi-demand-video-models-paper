import torch
import torch.nn as nn
from torch.optim import Adam
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader

import mlflow
import numpy as np

from prefect import flow, task

from conv_lstm import Seq2Seq
from datetime import datetime

mlflow_tracking_uri = 'http://localhost:5001'
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("OD-demand-models")


class AdjacencyMatrixDataset(Dataset):
    def __init__(self, adj_matrices, sequence_length=10):
        self.adj_matrices = adj_matrices
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.adj_matrices) - self.sequence_length + 1

    def __getitem__(self, idx):
        sequence = self.adj_matrices[idx:idx + self.sequence_length]
        sequence_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(1) # add channel dim
        return sequence_tensor

@task(log_prints=True)
def load_adj_matrices():
    print("Loading adjacency matrices...")
    loaded_data = np.load('adj_matrices.npz')
    adj_matrices = [loaded_data[key] for key in loaded_data]
    print(f"Loaded {len(adj_matrices)} adjacency matrices.")
    dataset = AdjacencyMatrixDataset(adj_matrices)

    train_size = int(0.8 * len(dataset))

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    for train_batch, test_batch in zip(train_loader, test_loader):
        print(train_batch.shape, test_batch.shape) # (batch_size, seq_len, channels, height, width)
        break

    return train_loader, test_loader

@flow(log_prints=True)
def train_conv_lstm():
    print("Starting training process...")
    train_loader, test_loader = load_adj_matrices()

    with mlflow.start_run():
    
        params = {
            "input_dim": 1,
            "hidden_dim": 64,
            "kernel_size": (3, 3),
            "num_layers": 5,
            "batch_first": True
        }
        print("Logging parameters to MLflow...")
        mlflow.log_params(params)
        
        lr = 1e-4
        mlflow.log_params({'lr': lr})
        device = 'mps'
        criterion = nn.MSELoss()
        model = Seq2Seq(**params).to(device)
        with open('model_summary.txt', 'w') as f:
            f.write(str(summary(model)))
        mlflow.log_artifact('model_summary.txt')
        
        optim = Adam(model.parameters(), lr=lr)

        num_epochs = 30

        for epoch in range(num_epochs):
            train_loss = 0
            for _, sequence in enumerate(train_loader):
                inputs = sequence[:, :-1, :, :, :].to(device)
                targets = sequence[:, 1:, :, :, :].to(device)

                outputs = model(inputs)
                if isinstance(outputs, list):
                    outputs = torch.stack(outputs)
                
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                
                optim.zero_grad()
                loss.backward()
                optim.step()

            loss = train_loss/len(train_loader)
            mlflow.log_metric("train_loss", loss, step=epoch)
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for _, sequence in enumerate(test_loader):
                    inputs = sequence[:, :-1, :, :, :].to(device)
                    targets = sequence[:, 1:, :, :, :].to(device)

                    outputs = model(inputs)
                    if isinstance(outputs, list):
                        outputs = torch.stack(outputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
            test_loss = test_loss/len(test_loader)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}')
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        torch.save(model.state_dict(), f"model_{current_time}.pth")
        mlflow.log_artifact(f"model_{current_time}.pth")

if __name__ == '__main__':
    train_conv_lstm()