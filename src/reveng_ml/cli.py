"""
Command-line interface
"""
import typer
from pathlib import Path
import torch

from reveng_ml.data import BinaryChunkDataset
from reveng_ml.model import get_model
from reveng_ml.trainer import Trainer
from reveng_ml.evaluate import Evaluator

app = typer.Typer(help="Function boundary detection model training & evaluation")

@app.command()
def train(
    data_dir: Path = typer.Option("data/train", "--data-dir", "-d", help="Training data input directory"),
    model_dir: Path = typer.Option("models", "--model-dir", "-o", help="Model output directory"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(5e-5, "--lr", "-l", help="Learning rate"),
    chunk_size: int = typer.Option(512, help="Size of each binary chunk"),
    stride: int = typer.Option(256, help="Stride for overlapping chunks"),
    class_weight_boundary: float = typer.Option(100.0, "--class-weight", "-w", help="Weight for boundary classes (B-FUNC, E-FUNC). Higher = more focus on boundaries"),
):
    """
    Train a new function boundary detection model.
    """
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"Error: Training data directory '{data_dir}' is empty or does not exist.")
        raise typer.Exit(code=1)

    # Load training data
    print(f"Loading data from {data_dir}...")
    dataset = BinaryChunkDataset(data_dir=data_dir, chunk_size=chunk_size, stride=stride)
    if not dataset:
        print("Warning: The dataset is empty. No training will be performed.")
        raise typer.Exit()
    print(f"Created dataset with {len(dataset)} chunks.")

    print("Initializing model...")
    model = get_model()

    # Train
    trainer = Trainer(model, dataset, learning_rate=learning_rate, batch_size=batch_size, model_dir=model_dir, class_weight_boundary=class_weight_boundary)
    trainer.train(epochs=epochs)

    # Save
    model_name = "reveng_boundary_detector_final.bin"
    trainer.save_model(model_name)
    print(f"Training complete. Model saved to {model_dir / model_name}.")

@app.command()
def evaluate(
    model_path: Path = typer.Option("models/reveng_boundary_detector_final.bin", "--model-path", "-m", help="Trained model path"),
    data_dir: Path = typer.Option("data/test", "--data-dir", "-d", help="Test data directory"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Evaluation batch size"),
    chunk_size: int = typer.Option(512, help="Size of each binary chunk"),
    stride: int = typer.Option(256, help="Stride for overlapping chunks"),
):
    """
    Evaluate a trained model on a test dataset.
    """
    print(f"Starting evaluation process...")

    if not model_path.exists():
        print(f"Error: Model file not found at '{model_path}'.")
        raise typer.Exit(code=1)
    
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"Error: Test data directory '{data_dir}' is empty or does not exist.")
        raise typer.Exit(code=1)

    # Initialize dataset
    print(f"Loading test data from {data_dir}...")
    dataset = BinaryChunkDataset(data_dir=data_dir, chunk_size=chunk_size, stride=stride)
    if not dataset:
        print("Warning: The test dataset is empty. No evaluation will be performed.")
        raise typer.Exit()
    print(f"Created test dataset with {len(dataset)} chunks.")

    # Load trained model
    print(f"Loading model from {model_path}...")
    model = get_model()
    model.load_state_dict(torch.load(model_path))

    # Evaluate
    evaluator = Evaluator(model, dataset, batch_size=batch_size)
    evaluator.evaluate()
    print(f"Evaluation complete.")

if __name__ == "__main__":
    app()
