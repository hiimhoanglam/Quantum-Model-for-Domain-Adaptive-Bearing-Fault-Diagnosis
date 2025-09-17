import torch
import os

# --- Paths ---
BASE_PATH = r"D:\ChungCheng_Project\FaultPrediction\CWRU-dataset-main-v3"  # Path to CWRU dataset v3
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")  # Directory for outputs (plots, models)
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create results dir if it doesn't exist

# --- Data Loading ---
SAMPLE_LENGTH = 1024  # Length of each data sample
PREPROCESSING = True  # Whether to apply preprocessing (e.g., cepstrum, envelope)
OVERLAPPING_RATIO = 0  # Overlap ratio for data segmentation
RANDOM_STATE = 42  # For reproducibility in data splits
NUM_CLASSES = 7  # Number of classes

# --- Model Parameters ---
N_QUBITS = 10  # Number of qubits for quantum circuit
SHOTS = None  # Number of shots for quantum simulation if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for PyTorch

# --- Training Parameters ---
NUM_RUNS = 3
BATCH_SIZE = 128  # Batch size for DataLoader
NUM_EPOCHS = 20  # Number of training epochs
LEARNING_RATE = 0.5  # Learning rate for optimizer
DROPOUT_RATE = 0.2  # Dropout probability
OPTIMIZER = "AdamW"  # Optimizer type
LOSS_FN = "CrossEntropyLoss"  # Loss function for classification

# --- Quantum Circuit Parameters ---
EMBEDDING_PARAMS = 10  # Number of params for embedding layer (from notebook)
VQC_PARAMS_PER_LAYER = 12  # Params per VQC layer
SEL_PARAMS_SHAPE = (1, N_QUBITS, 3)  # Shape for StronglyEntanglingLayers
TOTAL_NUM_PARAMS = (
    EMBEDDING_PARAMS
    + VQC_PARAMS_PER_LAYER * (N_QUBITS // 2)  # VQC1
    + VQC_PARAMS_PER_LAYER * (N_QUBITS // 2)  # VQC2
    + 1 * N_QUBITS * 3  # StronglyEntanglingLayers
)

# --- Evaluation and Visualization ---
METRICS = ["accuracy", "loss", "f1_score"]  # Metrics to track during training
PLOT_FORMATS = ["png"]  # File formats for saving plots
RESULTS_CSV = os.path.join(OUTPUT_DIR, "results.csv")  # Path for saving averaged results

# --- Logging ---
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")  # Directory for training logs
os.makedirs(LOG_DIR, exist_ok=True)

# --- Data Loading ---
DIAMETER_ORDER = ['7', '14', '21']  # Order for fault diameters; shuffle for different splits (e.g., ['14', '7', '21'])
PREPROCESSING_TYPE = 'envelope'  # Options: 'envelope', 'one_sided', 'none'
RANDOM_STATE = 42  # For reproducibility