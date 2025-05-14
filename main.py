from argparse import ArgumentParser
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from models.ae import train_ae_model, test_ae_model
from models.cmlp import train_cae_model, test_cae_model

def parse_args():
    parser = ArgumentParser(description="A script to detect fraud.")
    parser.add_argument("--input_file", type=Path, required=True, help="Path to the input CSV file (must contain a 'Fraud' column).")
    parser.add_argument("--train", action="store_true", help="Whether to train the model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--model_type", choices=["ae", "cae"], default="ae", help="Model type: 'ae' for Autoencoder, 'cae' for Convolutional Autoencoder.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: 8 for AE, 16 for CAE).")
    parser.add_argument("-o", "--output_name", type=str, default="model.pth", help="Name of the output model file.")
    return parser.parse_args()

def ensure_output_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def run_training(df: pd.DataFrame, model_type: str, epochs: int, batch_size: int, device: str, output_path: Path):
    if model_type == "ae":
        # Entreno solo con muestras no fraudulentas
        no_fraud = df[df["Fraud"] == 0].drop(columns="Fraud").values
        train_data, temp = train_test_split(no_fraud, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp, test_size=0.5, random_state=42)
        hidden_dim = 4
        bs = batch_size or 8
        model = train_ae_model(train_data, val_data,
                               num_epochs=epochs,
                               batch_size=bs,
                               hidden_dim=hidden_dim,
                               device=device)
    else:
        # con todo menos miedo
        all_data = df.drop(columns="Fraud").values
        train_data, temp = train_test_split(all_data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp, test_size=0.5, random_state=42)
        hidden_dim = 4
        bs = batch_size or 16
        model = train_cae_model(train_data, val_data,
                                num_epochs=epochs,
                                batch_size=bs,
                                hidden_dim=hidden_dim,
                                device=device)

    ensure_output_dir(output_path)
    torch.save(model, output_path)
    print(f"Model saved to {output_path}")

    
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    if model_type == "ae":
        test_loss = test_ae_model(model, test_tensor, device)
    else:
        test_loss = test_cae_model(model, test_tensor, device)
    print(f"Test Loss: {test_loss:.4f}")

    return model


def run_evaluation(df: pd.DataFrame, model, model_type: str, device: str):
    fraud = df[df["Fraud"] == 1].drop(columns="Fraud").values
    if model_type == "ae":
        no_fraud = df[df["Fraud"] == 0].drop(columns="Fraud").values
        _, temp = train_test_split(no_fraud, test_size=0.3, random_state=42)
        _, test_data = train_test_split(temp, test_size=0.5, random_state=42)
        combined = np.vstack([test_data, fraud])
    else:
        all_data = df.drop(columns="Fraud").values
        _, temp = train_test_split(all_data, test_size=0.3, random_state=42)
        _, test_data = train_test_split(temp, test_size=0.5, random_state=42)
        combined = np.vstack([test_data, fraud])

    labels = np.array([0] * len(test_data) + [1] * len(fraud))
    tensor = torch.tensor(combined, dtype=torch.float32).to(device)

    # errores de reconstrucción
    if model_type == "ae":
        errors = test_ae_model(model, tensor, device)
    else:
        errors = test_cae_model(model, tensor, device)

    # Umbral al percentil 95
    threshold = np.percentile(errors[:len(test_data)], 95)
    print(f"Threshold for anomaly detection (95th percentile): {threshold:.4f}")

    preds = (errors > threshold).astype(int)

    # Métricas
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, digits=4)

    print("Test Evaluation:")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.input_file)

    model_path = Path("./outputs/model") / args.output_name

    if args.train:
        model = run_training(
            df=df,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size or 0,
            device=device,
            output_path=model_path
        )
    else:
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()

    run_evaluation(
        df=df,
        model=model,
        model_type=args.model_type,
        device=device
    )


if __name__ == "__main__":
    main()
