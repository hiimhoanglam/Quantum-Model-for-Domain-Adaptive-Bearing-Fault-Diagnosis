import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from src.data.data_loader import data_import, normalize_data
from src.models.quantum_model import QuantumModel
from config import (
    BASE_PATH, SAMPLE_LENGTH, PREPROCESSING, OVERLAPPING_RATIO,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE,
    OPTIMIZER, LOSS_FN, METRICS, PLOT_FORMATS, OUTPUT_DIR,
    LOG_DIR, RESULTS_CSV
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_optimizer_and_loss(model):
    """Initialize optimizer and loss function based on config."""
    if OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")
    
    if LOSS_FN == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {LOSS_FN}")
    
    return optimizer, criterion

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, dtype=torch.float64), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1

def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on validation/test data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device, dtype=torch.float64), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / total
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, acc, f1

def save_plot(history, metric, output_dir, formats):
    """Save training and validation metric plots."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train'][metric], label=f'Train {metric.capitalize()}')
    plt.plot(history['val'][metric], label=f'Validation {metric.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} over Epochs')
    plt.legend()
    plt.grid(True)
    
    for fmt in formats:
        plt.savefig(os.path.join(output_dir, f'{metric}_plot.{fmt}'))
    plt.close()

def main():
    """Main training loop."""
    logger.info("Starting training process...")
    
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
        
        # Log dataset shapes
        logger.info(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
        logger.info(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
        
        # Adjust input dimension based on PREPROCESSING_TYPE
        input_dim = X_train.shape[1]
        logger.info(f"Input dimension: {input_dim}")
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float64)
        Y_train = torch.tensor(Y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float64)
        Y_val = torch.tensor(Y_val, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float64)
        Y_test = torch.tensor(Y_test, dtype=torch.long)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)
        test_dataset = TensorDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        logger.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    # Initialize model, optimizer, and loss
    try:
        model = QuantumModel().to(DEVICE)
        optimizer, criterion = get_optimizer_and_loss(model)
        logger.info(f"Model initialized on {DEVICE}. Optimizer: {OPTIMIZER}, Loss: {LOSS_FN}")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise
    
    # Training loop
    history = {'train': {m: [] for m in METRICS}, 'val': {m: [] for m in METRICS}}
    best_val_f1 = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    
    for epoch in range(NUM_EPOCHS):
        try:
            train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, DEVICE)
            
            history['train']['loss'].append(train_loss)
            history['train']['accuracy'].append(train_acc)
            history['train']['f1_score'].append(train_f1)
            history['val']['loss'].append(val_loss)
            history['val']['accuracy'].append(val_acc)
            history['val']['f1_score'].append(val_f1)
            
            logger.info(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
            )
            
            # Save best model based on validation F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model with Val F1: {best_val_f1:.4f}")
        except Exception as e:
            logger.error(f"Error in epoch {epoch+1}: {str(e)}")
            raise
    
    # Evaluate on test set
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, DEVICE)
        logger.info(
            f"Test Results | Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}"
        )
        
        # Save results to CSV
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1_score': test_f1,
            'best_val_f1': best_val_f1
        }
        pd.DataFrame([results]).to_csv(RESULTS_CSV, index=False)
        logger.info(f"Saved results to {RESULTS_CSV}")
    except Exception as e:
        logger.error(f"Error evaluating test set or saving results: {str(e)}")
        raise
    
    # Save plots
    try:
        for metric in METRICS:
            save_plot(history, metric, OUTPUT_DIR, PLOT_FORMATS)
            logger.info(f"Saved {metric} plot to {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Error saving plots: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise