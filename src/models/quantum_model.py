import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from config import N_QUBITS, NUM_CLASSES, SAMPLE_LENGTH, DROPOUT_RATE, SHOTS, EMBEDDING_PARAMS, VQC_PARAMS_PER_LAYER

# Device setup
dev = qml.device("default.qubit", wires=N_QUBITS, shots=SHOTS)

def state_preparation(features):
    """Prepare input features using AmplitudeEmbedding."""
    qml.AmplitudeEmbedding(features, wires=range(N_QUBITS), normalize=True)

def embedding_layer(params):
    """Apply embedding layer with Hadamard and RY gates."""
    # params shape: (EMBEDDING_PARAMS,)
    for i in range(N_QUBITS):
        qml.H(wires=i)
    for i in range(N_QUBITS):
        qml.RY(params[i], wires=i)

def vqc(params, index):
    """Main Variational Quantum Circuit (VQC) layer."""
    # params shape: (VQC_PARAMS_PER_LAYER,)
    qml.RX(params[0], wires=index)
    qml.RY(params[1], wires=index + 1)
    qml.RY(params[2], wires=index)
    qml.RZ(params[3], wires=index + 1)
    qml.RZ(params[4], wires=index)
    qml.RX(params[5], wires=index + 1)
    qml.CZ(wires=[index, index + 1])
    qml.RZ(params[6], wires=index)
    qml.RX(params[7], wires=index + 1)
    qml.RY(params[8], wires=index)
    qml.RZ(params[9], wires=index + 1)
    qml.RX(params[10], wires=index)
    qml.RY(params[11], wires=index + 1)
    qml.CNOT(wires=[index + 1, index])

@qml.qnode(dev, interface="torch")
def full_circuit(inputs, all_params):
    """Full quantum circuit combining embedding and VQC layers."""
    embed_len = EMBEDDING_PARAMS
    vqc1_len = VQC_PARAMS_PER_LAYER * (N_QUBITS // 2)  # 60
    vqc2_len = VQC_PARAMS_PER_LAYER * (N_QUBITS // 2)  # another 60
    sel_len = embed_len + vqc1_len + vqc2_len

    embedding_params = all_params[:embed_len]
    vqc1_params = all_params[embed_len:embed_len + vqc1_len]
    vqc2_params = all_params[embed_len + vqc1_len:sel_len]
    sel_params = all_params[sel_len:]

    vqc1_params = vqc1_params.view(N_QUBITS // 2, VQC_PARAMS_PER_LAYER)
    vqc2_params = vqc2_params.view(N_QUBITS // 2, VQC_PARAMS_PER_LAYER)
    sel_params = sel_params.view(1, N_QUBITS, 3)

    state_preparation(inputs)
    embedding_layer(embedding_params)

    for i in range(N_QUBITS // 2):
        vqc(vqc1_params[i], i * 2)
    
    for i in range(N_QUBITS // 2):
        vqc(vqc2_params[i], i * 2)
        
    for i in range(N_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    qml.StronglyEntanglingLayers(weights=sel_params, wires=range(N_QUBITS))

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Total number of parameters
total_num_params = (
    EMBEDDING_PARAMS
    + VQC_PARAMS_PER_LAYER * (N_QUBITS // 2)  # VQC1
    + VQC_PARAMS_PER_LAYER * (N_QUBITS // 2)  # VQC2
    + 1 * N_QUBITS * 3  # StronglyEntanglingLayers
)
weight_shapes = {"all_params": total_num_params}

class QuantumModel(nn.Module):
    """PyTorch wrapper for the quantum circuit."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(SAMPLE_LENGTH, N_QUBITS, dtype=torch.float64)
        self.q_layer = qml.qnn.TorchLayer(full_circuit, weight_shapes)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.fc = nn.Linear(N_QUBITS, NUM_CLASSES, dtype=torch.float64)

    def forward(self, x):
        x_proj = self.input_proj(x)  # Shape: [batch_size, N_QUBITS]
        q_out = self.q_layer(x)      # Shape: [batch_size, N_QUBITS]
        q_out = q_out + x_proj.to(q_out.device)  # Residual connection
        q_out = self.dropout(q_out)
        logits = self.fc(q_out)
        return logits