"""
Training script for the RevEng-ML project.
"""
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from reveng_ml.utils import get_pytorch_device


class Trainer:
    """Trains a model using a dataset"""

    def __init__(self, model, dataset, learning_rate=5e-5, batch_size=32, model_dir=Path('./models')):
        """
        Create a new Trainer class.

        Args:
            model: PyTorch model
            dataset: PyTorch dataset
            learning_rate (float): Optimizer learning rate
            batch_size (int): Samples per batch
            model_dir (Path): Model output directory
        """
        self.device = get_pytorch_device()
        self.model = model.to(self.device)
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)

    def train(self, epochs=3):
        """
        Train for a number of epochs.

        Args:
            epochs (int): Epoch count
        """
        self.model.train()
        
        for epoch in range(epochs):
            print(f"--- Starting Epoch {epoch + 1}/{epochs} ---")
            epoch_start_time = time.time()
            total_loss = 0
            
            # Wrap with tqdm() to show progress_bar
            progress_bar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

            for i, (batch_data, batch_labels) in enumerate(progress_bar):
                batch_labels = batch_labels.to(self.device)

                # Clear prev. gradients
                self.model.zero_grad()

                # Forward pass
                outputs = self.model(input_ids=batch_data, labels=batch_labels)
                
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Add current loss behind progressbar
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / len(self.loader)
            epoch_time = time.time() - epoch_start_time
            print(f"--- Epoch {epoch + 1} Summary ---")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Epoch Time: {epoch_time:.2f} seconds")
            print("-" * (25 + len(str(epoch+1))))


    def save_model(self, filename="reveng_model.bin"):
        """Saves the model state"""
        save_path = self.model_dir / filename
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")