from argparse import ArgumentParser
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from models.ae import train_ae_model, test_ae_model
from models.cmlp import train_cae_model, test_cae_model
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def parse_args():
    parser = ArgumentParser(description="A script to detect fraud.")
    parser.add_argument("--input_file", type=Path, help="Path to the input file to be processed.")
    train_args = parser.add_argument_group("Training arguments")
    train_args.add_argument("--train", action="store_true", help="Whether to train the model or not.")
    train_args.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--model_type", choices=["ae", "cae"], default="ae", help="Model type: 'ae' for Autoencoder, 'cae' for Convolutional Autoencoder.")
    parser.add_argument("-o", "--output_name", type=str, default="model.pth", help="Name of the output model file.")
    return parser.parse_args()

def main():
    args = parse_args()
    input_file = args.input_file
    train = args.train
    num_epochs = args.epochs
    model_type = args.model_type
    output_name = args.output_name  # Output model name

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    
    data = pd.read_csv(input_file)

    if train:
        if model_type == "ae":
            data_fraud = data[data["Fraud"] == 1].drop(columns=["Fraud"]).values
            data_no_fraud = data[data["Fraud"] == 0].drop(columns=["Fraud"]).values
            
            # Split data
            train_data, temp_data = train_test_split(data_no_fraud, test_size=0.3, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
            model = train_ae_model(train_data, val_data, num_epochs=num_epochs, batch_size=8, hidden_dim=4, device=device)
        else:  # model_type == "cae"
            train_data, temp_data = train_test_split(data.values, test_size=0.3, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
            model = train_cae_model(train_data, val_data, num_epochs=num_epochs, batch_size=16, hidden_dim=4, device=device)

        output_path = f"./outputs/model/{output_name}"
        torch.save(model, output_path)
        print(f"Model saved to {output_path}")

        # Test model
        test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
        if model_type == "ae":
            test_loss = test_ae_model(model, test_tensor, device)
        else:
            test_loss = test_cae_model(model, test_data, device)
        print(f"Test Loss: {test_loss}")
    else:
        output_path = f"./outputs/model/{output_name}"
        model = torch.load(output_path, map_location=device)
        model.to(device)
    
    model.eval()

    # Get reconstruction errors
    if model_type == "ae":
        data_fraud = data[data["Fraud"] == 1].drop(columns=["Fraud"]).values
        data_no_fraud = data[data["Fraud"] == 0].drop(columns=["Fraud"]).values

        train_data, temp_data = train_test_split(data_no_fraud, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        test_combined = np.vstack([test_data, data_fraud])
        test_labels = np.array([0.0] * len(test_data) + [1.0] * len(data_fraud))
        
        test_tensor = torch.tensor(test_combined, dtype=torch.float32).to(device)
        errors = test_ae_model(model, test_tensor, device)
        
    else:
        data_fraud = data[data["Fraud"] == 1].values
        data_no_fraud = data[data["Fraud"] == 0].values
        
        train_data, temp_data = train_test_split(data.values, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        test_combined = np.vstack([test_data, data_fraud])
    
        test_labels = np.array([0.0] * len(test_data) + [1.0] * len(data_fraud))
        test_tensor = torch.tensor(test_combined).to(device)
        errors = test_cae_model(model, test_tensor, device) # witj BCE
        
    threshold = np.percentile(errors[:len(test_data)], 95)
    print(f"Threshold for anomaly detection: {threshold}")
    preds = (errors > threshold).astype(int)

    conf = confusion_matrix(test_labels, preds)
    report = classification_report(test_labels, preds, digits=4)

    print("Test Evaluation:")
    print("Confusion Matrix:\n", conf)
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    main()