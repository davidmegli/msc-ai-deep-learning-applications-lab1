
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm

from models.model import ResidualCNN

def extract_features(model, dataloader, device):
    """ Extracts features from the model for the given dataloader.
    Args:
        model (nn.Module): The model from which to extract features.
        dataloader (DataLoader): The dataloader for the dataset.
        device (torch.device): The device to run the model on (CPU or GPU).
    Returns:
        tuple: A tuple containing the extracted features and their corresponding labels.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            features = model(inputs, return_features=True)
            all_features.append(features.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_features, all_labels

def main(args):
    """ Main function to extract features from CIFAR-100 dataset using a trained ResidualCNN model.
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset CIFAR-100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # Modello
    model = ResidualCNN(num_classes=10, depth=32, width=64)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print("Extracting features...")
    features, labels = extract_features(model, dataloader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "cifar100_features.npz")
    np.savez(output_path, features=features, labels=labels)
    print(f"Features saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained ResidualCNN on CIFAR-10")
    parser.add_argument("--output_dir", type=str, default="outputs/features", help="Directory to save features")
    args = parser.parse_args()

    main(args)