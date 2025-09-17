import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from src.data.data_loader import data_import, normalize_data
from src.models.quantum_model import QuantumModel
from config import (
    BASE_PATH, SAMPLE_LENGTH, PREPROCESSING, OVERLAPPING_RATIO,
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    PLOT_FORMATS, OUTPUT_DIR, LOG_DIR, NUM_RUNS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f'average_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_data_loaders(X, Y, batch_size, shuffle=False):
    """Create a DataLoader from numpy arrays."""
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float64),
        torch.tensor(Y, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate the model on a DataLoader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device, dtype=torch.float64), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, acc, f1

def train_and_evaluate(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size, num_epochs, num_runs, device, early_stop_patience=10):
    """Train and evaluate the model over multiple runs with early stopping."""
    val_loader = create_data_loaders(X_val, Y_val, batch_size)
    test_loader = create_data_loaders(X_test, Y_test, batch_size)
    
    all_train_losses, all_val_losses, all_test_losses = [], [], []
    all_train_accs, all_val_accs, all_test_accs = [], [], []
    all_train_f1s, all_val_f1s, all_test_f1s = [], [], []
    
    for run in range(num_runs):
        logger.info(f"Starting run {run+1}/{num_runs}")
        torch.manual_seed(run)
        train_loader = create_data_loaders(X_train, Y_train, batch_size, shuffle=True)
        
        try:
            model = QuantumModel().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            # Early stopping variables
            best_val_loss = float("inf")
            best_state = None
            epochs_no_improve = 0
            
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device, dtype=torch.float64), labels.to(device)
                    optimizer.zero_grad()
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                # Evaluate after each epoch
                train_loss, train_acc, train_f1 = evaluate_model(model, train_loader, criterion, device)
                val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
                
                logger.info(
                    f"Run {run+1}, Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
                )
                
                # Early stopping on val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stop_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs in run {run+1}")
                        model.load_state_dict(best_state)
                        break
            
            # Final test evaluation
            test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion, device)
            logger.info(
                f"Run {run+1} Test Results | Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}"
            )
            
            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)
            all_test_losses.append(test_loss)
            all_train_accs.append(train_acc)
            all_val_accs.append(val_acc)
            all_test_accs.append(test_acc)
            all_train_f1s.append(train_f1)
            all_val_f1s.append(val_f1)
            all_test_f1s.append(test_f1)
            
        except Exception as e:
            logger.error(f"Error in run {run+1}: {str(e)}")
            raise
    
    return all_train_losses, all_val_losses, all_test_losses, all_train_accs, all_val_accs, all_test_accs, all_train_f1s, all_val_f1s, all_test_f1s

def save_plot(history, metric, output_dir, formats):
    """Save metric plots for multiple runs."""
    plt.figure(figsize=(10, 6))
    plt.plot(history, label=f'{metric.capitalize()} per Run')
    plt.xlabel('Run')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} Across Runs')
    plt.legend()
    plt.grid(True)
    
    for fmt in formats:
        plt.savefig(os.path.join(output_dir, f'{metric}_runs_plot.{fmt}'))
    plt.close()

def main():
    """Main function to load data and run multiple training evaluations."""
    logger.info("Starting average results computation...")
    
    # Verify dataset path
    if not os.path.exists(BASE_PATH):
        logger.error(f"Dataset path does not exist: {BASE_PATH}")
        raise FileNotFoundError(f"Dataset path does not exist: {BASE_PATH}")
    
    # Load and normalize data
    try:
        logger.info("Loading CWRU dataset...")
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data_import(
            base_path=BASE_PATH,
            sample_length=SAMPLE_LENGTH,
            preprocessing=PREPROCESSING,
            overlapping_ratio=OVERLAPPING_RATIO
        )
        
        if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
            logger.error("One or more datasets are empty")
            raise ValueError("One or more datasets are empty")
        
        logger.info("Normalizing data...")
        X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
        
        logger.info(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
        logger.info(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    # Train and evaluate over multiple runs
    try:
        metrics = train_and_evaluate(
            X_train, Y_train, X_val, Y_val, X_test, Y_test,
            batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, num_runs=NUM_RUNS,
            device=DEVICE, early_stop_patience=10
        )
        all_train_losses, all_val_losses, all_test_losses, all_train_accs, all_val_accs, all_test_accs, all_train_f1s, all_val_f1s, all_test_f1s = metrics
        
        # Compute and log average results
        results = {
            'train_loss_mean': np.mean(all_train_losses),
            'train_loss_std': np.std(all_train_losses),
            'val_loss_mean': np.mean(all_val_losses),
            'val_loss_std': np.std(all_val_losses),
            'test_loss_mean': np.mean(all_test_losses),
            'test_loss_std': np.std(all_test_losses),
            'train_acc_mean': np.mean(all_train_accs),
            'train_acc_std': np.std(all_train_accs),
            'val_acc_mean': np.mean(all_val_accs),
            'val_acc_std': np.std(all_val_accs),
            'test_acc_mean': np.mean(all_test_accs),
            'test_acc_std': np.std(all_test_accs),
            'train_f1_mean': np.mean(all_train_f1s),
            'train_f1_std': np.std(all_train_f1s),
            'val_f1_mean': np.mean(all_val_f1s),
            'val_f1_std': np.std(all_val_f1s),
            'test_f1_mean': np.mean(all_test_f1s),
            'test_f1_std': np.std(all_test_f1s)
        }
        
        logger.info("\n=== Final Average Results after %d Runs ===", NUM_RUNS)
        logger.info(f"Train Loss: {results['train_loss_mean']:.4f} ± {results['train_loss_std']:.4f}")
        logger.info(f"Val Loss:   {results['val_loss_mean']:.4f} ± {results['val_loss_std']:.4f}")
        logger.info(f"Test Loss:  {results['test_loss_mean']:.4f} ± {results['test_loss_std']:.4f}")
        logger.info(f"Train Acc:  {results['train_acc_mean']:.4f} ± {results['train_acc_std']:.4f}")
        logger.info(f"Val Acc:    {results['val_acc_mean']:.4f} ± {results['val_acc_std']:.4f}")
        logger.info(f"Test Acc:   {results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}")
        logger.info(f"Train F1:   {results['train_f1_mean']:.4f} ± {results['train_f1_std']:.4f}")
        logger.info(f"Val F1:     {results['val_f1_mean']:.4f} ± {results['val_f1_std']:.4f}")
        logger.info(f"Test F1:    {results['test_f1_mean']:.4f} ± {results['test_f1_std']:.4f}")
        
        # Save results to CSV
        results_df = pd.DataFrame([results])
        results_csv_path = os.path.join(OUTPUT_DIR, 'average_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Saved average results to {results_csv_path}")
        
        # Save plots for metrics across runs
        history = {
            'train_loss': all_train_losses,
            'val_loss': all_val_losses,
            'test_loss': all_test_losses,
            'train_acc': all_train_accs,
            'val_acc': all_val_accs,
            'test_acc': all_test_accs,
            'train_f1': all_train_f1s,
            'val_f1': all_val_f1s,
            'test_f1': all_test_f1s
        }
        for metric in ['train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc', 'train_f1', 'val_f1', 'test_f1']:
            save_plot(history[metric], metric, OUTPUT_DIR, PLOT_FORMATS)
            logger.info(f"Saved {metric} plot to {OUTPUT_DIR}")
            
    except Exception as e:
        logger.error(f"Error in training/evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise