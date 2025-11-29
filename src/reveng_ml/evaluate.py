"""
Evaluation script for the RevEng-ML project.
"""
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from reveng_ml.utils import get_pytorch_device


class Evaluator:
    """Evaluates a trained model"""

    def __init__(self, model, dataset, batch_size=32):
        """
        Creates a new Evaluator class

        Args:
            model: Trained PyTorch model to evaluate
            dataset: PyTorch dataset
            batch_size (int): Batch size for evaluation
        """
        self.device = get_pytorch_device()
        self.model = model.to(self.device)
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def evaluate(self):
        """
        Execute evaluation

        Returns:
            A dictionary containing the classification report from scikit-learn
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        print("Starting evaluation...")
        progress_bar = tqdm(self.loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch_data, batch_labels in progress_bar:
                batch_data = batch_data.to(self.device)
                
                # Get model predictions
                outputs = self.model(input_ids=batch_data)
                logits = outputs.logits
                
                # class with the highest score
                predictions = torch.argmax(logits, dim=-1).cpu().numpy().flatten()
                
                all_preds.extend(predictions)
                all_labels.extend(batch_labels.cpu().numpy().flatten())

        print("Evaluation complete.")
        
        # Print a classification report
        report = classification_report(
            all_labels,
            all_preds,
            # O = None, B-FUNC = Beginning of a function, E-FUNC = End of a function
            target_names=['O', 'B-FUNC', 'E-FUNC'],
            zero_division=0
        )
        
        print("\n--- Classification Report ---")
        print(report)
        print("-----------------------------\n")

        return report
