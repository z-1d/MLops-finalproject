import os
import sys

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# 引入项目根目录以正常 import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.model import WineNet


class WineDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        self.X = torch.tensor(df.drop("target", axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(df["target"].values, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(data_path, epochs=50, lr=0.01):
    dataset = WineDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = WineNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    mlflow.set_experiment("Wine_PyTorch_Experiment")
    with mlflow.start_run():
        mlflow.log_param("dataset_version", data_path)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            outputs = model(dataset.X)
            _, preds = torch.max(outputs, 1)
            acc = (preds == dataset.y).float().mean().item()
        
        mlflow.log_metric("accuracy", acc)
        
        os.makedirs("ml/registry", exist_ok=True)
        model_path = "ml/registry/model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        print(f"Trained on {data_path} | Epochs: {epochs} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/v1_wine.csv"
    eps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    train(data_file, epochs=eps)
