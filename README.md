# Quantum Model for Domain-Adaptive Bearing Fault Diagnosis

This repository implements a quantum machine learning model for domain-adaptive bearing fault diagnosis using the CWRU bearing dataset. The project a novel structure of parameterized quantum circuit and classical components integration such as residual connection and dropout to classify bearing faults across different machine conditions.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Files](#key-files)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
The project develops a quantum-classical hybrid model for bearing fault diagnosis, addressing challenges in domain adaptation. It uses the Case Western Reserve University (CWRU) bearing dataset, preprocessed with envelope analysis or one-sided spectra, to train models that classify faults (Normal, Ball, Inner Race, Outer Race) across Drive End (DE) and Fan End (FE) bearings. The quantum model is implemented using PennyLane, with comparisons to classical models like WDCNN and ACDIN.

## Repository Structure
```
FaultPrediction/
├── .gitignore                  # Git ignore rules
├── config.py                   # Configuration settings (e.g., dataset path, preprocessing)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── __init__.py                 # Makes FaultPrediction a Python package
├── CWRU-dataset-main-v3/
│   └── CWRU-dataset-main-v3.rar  # CWRU dataset (via Git LFS)
├── src/
│   ├── data/
│   │   ├── data_loader.py     # Data loading and preprocessing
│   │   └── __init__.py
│   ├── models/
│   │   ├── acdin.py           # ACDIN classical model
│   │   ├── quantum_model.py   # Proposed quantum model
│   │   ├── wdcnn.py           # WDCNN classical model
│   │   └── __init__.py
│   ├── utils/
│   │   ├── metrics.py         # Evaluation metrics (placeholder)
│   │   ├── visualization.py   # Visualization utilities (placeholder)
│   │   └── __init__.py
│   └── __init__.py
├── scripts/
│   ├── average_results.py     # Aggregate results across runs
│   ├── load_data.py          # Load and preprocess CWRU dataset
│   ├── train_single.py       # Train a single model
│   ├── noise_simulation.py   # Simulate noise (TODO)
│   └── __init__.py
```

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/hiimhoanglam/Quantum-Model-for-Domain-Adaptive-Bearing-Fault-Diagnosis.git
   cd Quantum-Model-for-Domain-Adaptive-Bearing-Fault-Diagnosis
   ```

2. **Install Git LFS** (for the dataset):
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   - `pennylane`
   - `torch`
   - `numpy`
   - `matplotlib`
   - `scipy`
   - `pandas`

5. **Verify Dataset**:
   Ensure `CWRU-dataset-main-v3/CWRU-dataset-main-v3.rar` is downloaded (via Git LFS) and extracted to `CWRU-dataset-main-v3/`. Update `BASE_PATH` in `config.py` to point to the extracted dataset directory (e.g., `D:\ChungCheng_Project\FaultPrediction\CWRU-dataset-main-v3`).

## Usage
1. **Load and Preprocess Data**:
   Run the data loading script to prepare the CWRU dataset:
   ```bash
   python -m scripts.load_data
   ```
   This loads the dataset, splits it into train/val/test sets based on fault diameters (configured in `config.py`), and applies preprocessing (envelope analysis or one-sided spectra).

2. **Train the Model**:
   Train the quantum model or other models (e.g., WDCNN, ACDIN):
   ```bash
   python -m scripts.train_single
   ```
   Modify `config.py` to adjust parameters like `SAMPLE_LENGTH`, `PREPROCESSING_TYPE` (`envelope`, `one_sided`, `none`), or `DIAMETER_ORDER`.

3. **Aggregate Results**:
   Combine results from multiple runs:
   ```bash
   python -m scripts.average_results
   ```

4. **Configuration**:
   Edit `config.py` to customize:
   - `BASE_PATH`: Path to the CWRU dataset.
   - `SAMPLE_LENGTH`: Length of each data sample.
   - `PREPROCESSING`: Enable/disable preprocessing.
   - `PREPROCESSING_TYPE`: Choose `envelope`, `one_sided`, or `none`.
   - `DIAMETER_ORDER`: Fault diameter split (e.g., `['7', '14', '21']`).
   - `OVERLAPPING_RATIO`: Overlap for training data segmentation.

## Key Files
- **`config.py`**: Central configuration file for dataset paths, model parameters, and preprocessing options.
- **`src/data/data_loader.py`**: Loads and preprocesses the CWRU dataset, splitting by fault diameters.
- **`src/models/quantum_model.py`**: Quantum model implementation using PennyLane.
- **`src/models/wdcnn.py`**: Classical WDCNN model for comparison.
- **`src/models/acdin.py`**: ACDIN model for domain adaptation.
- **`scripts/load_data.py`**: Script to load and preprocess data.
- **`scripts/train_single.py`**: Script to train a single model instance.
- **`scripts/average_results.py`**: Aggregates results across multiple runs.
- **`CWRU-dataset-main-v3.rar`**: CWRU dataset (via Git LFS).

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure code follows PEP 8 style guidelines and includes tests where applicable.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details (to be added).

## Contact
For questions or issues, contact [hiimhoanglam](https://github.com/hiimhoanglam) or open an issue on GitHub.