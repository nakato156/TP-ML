import torch
import numpy as np

def test_ae_model(model, data: torch.Tensor, device: str = "cuda") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        recon = model(data)
        errors = torch.mean((data - recon) ** 2, dim=1)
    return errors.cpu().numpy()