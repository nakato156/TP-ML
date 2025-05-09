import torch
import torch.nn as nn
import pandas as pd
from .ae import AutoEncoder

features_names = ["Amount", "MCC", "TransactionSpeed"]

def train_ae_model(data: pd.DataFrame, num_epochs=100, batch_size=32, hidden_dim=64):
    features = data.loc[:, features_names]
    model = AutoEncoder(input_dim=len(features_names), hidden_dim=hidden_dim)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(features.values, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch[0])
            loss = criterion(outputs, batch[0])
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return model

