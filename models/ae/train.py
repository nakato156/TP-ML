import random
from pathlib import Path
from datetime import datetime

from .ae import AutoEncoder

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train_ae_model(train_data: np.ndarray, val_data: np.ndarray, num_epochs: int = 100, batch_size: int = 64, hidden_dim: int = 4, lr: float = 1e-3, device: str = "cuda") -> AutoEncoder:
    log_dir = Path("outputs") / "report" / f"AE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    model = AutoEncoder(input_dim=train_data.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor),
        batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=4
    )

    val_tensor = None
    if val_data is not None:
        val_tensor = torch.tensor(val_data, dtype=torch.float32).to(device)

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch", leave=False):
        model.train()
        epoch_loss = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)

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
                recon_val = model(val_tensor)
                val_loss = criterion(recon_val, val_tensor).item()
                writer.add_scalar("Loss/Validation", val_loss, epoch)

        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(f"Epoch {epoch}/{num_epochs} \tTrain Loss: {epoch_loss:.4f}\tValidation Loss: {val_loss:.4f}" if val_loss is not None else f"Epoch {epoch}/{num_epochs} \tTrain Loss: {epoch_loss:.4f}")

    writer.close()
    tqdm.write("[*] Training complete.")
    return model
