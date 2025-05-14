import random
from pathlib import Path
from datetime import datetime

from .mclp import CMLP

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train_model(train_data: np.ndarray, val_data: np.ndarray, num_epochs: int = 100, batch_size: int = 64, hidden_dim: int = 4, lr: float = 1e-3, device: str = "cuda"):
    log_dir = Path("outputs") / "report" / f"CAE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    X = train_data[:, :-1]
    Y = train_data[:, -1]
    
    model = CMLP(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)  # 2 clases: fraude/no fraude
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_tensor = torch.tensor(X, dtype=torch.float32)
    labels_tensor = torch.tensor(Y, dtype=torch.long)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor, labels_tensor),
        batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=4
    )

    val_tensor = None
    if val_data is not None:
        val_tensor = torch.tensor(val_data[:, :-1], dtype=torch.float32).to(device)
        val_labels = torch.tensor(val_data[:, -1], dtype=torch.long).to(device)

    criterion = nn.BCELoss()
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch", leave=False):
        model.train()
        epoch_loss = 0.0

        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1, 1).float()

            class_output = model(batch)
            loss = criterion(class_output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(train_tensor)
        writer.add_scalar("Loss/Train", epoch_loss, epoch)

        val_loss = None
        if val_tensor is not None:
            model.eval()
            with torch.no_grad():
                class_output = model(val_tensor)
                val_loss = criterion(class_output, val_labels.reshape(-1, 1).float()).item()
                writer.add_scalar("Loss/Validation", val_loss, epoch)

        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(f"Epoch {epoch}/{num_epochs} \tTrain Loss: {epoch_loss:.4f}\tValidation Loss: {val_loss:.4f}" if val_loss is not None else f"Epoch {epoch}/{num_epochs} \tTrain Loss: {epoch_loss:.4f}")

    writer.close()
    tqdm.write("[*] Training complete.")
    return model
