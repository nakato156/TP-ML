import torch
import torch.nn.functional as F

def test_model(model, test_data, device):
    model.eval()
    X = test_data[:, :-1]
    Y = test_data[:, -1].reshape(-1, 1)
    test_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(test_tensor)
    return outputs.reshape(-1, 1).cpu().numpy()