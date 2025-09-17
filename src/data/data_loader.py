import os
import numpy as np
import random
import torch
from scipy.signal import hilbert
from scipy.fft import fft, ifft
import scipy
from scipy.signal import resample
from config import RANDOM_STATE, BASE_PATH, SAMPLE_LENGTH, PREPROCESSING, OVERLAPPING_RATIO, DIAMETER_ORDER, PREPROCESSING_TYPE

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_STATE)

def analyze_data(Y):
    """Analyze class distribution in the dataset."""
    if len(Y) == 0:
        print('Error: No data')
        return
    unique_classes, counts = np.unique(Y, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(Y)) * 100
        print(f' - Class {cls}: {count} samples ({percentage:.2f}%)')

def get_fault_label(fault_type, bearing):
    """
    Return label based on fault type and bearing location:
    - 0 = Normal
    - 1 = Ball @ DE
    - 2 = Ball @ FE
    - 3 = IR @ DE
    - 4 = IR @ FE
    - 5 = OR @ DE
    - 6 = OR @ FE
    """
    fault_map = {
        (0, 'DE'): 1, (0, 'FE'): 2,
        (1, 'DE'): 3, (1, 'FE'): 4,
        (2, 'DE'): 5, (2, 'FE'): 6,
    }
    return fault_map.get((fault_type, bearing), ValueError(f"Invalid fault_type {fault_type} or bearing {bearing}"))

def import_file_specific(bearing, fault_type, diameter, sample_length, base_path, print_infor=False, overlapping_ratio=0.0):
    """
    Load specific fault data from .mat files.

    For OR faults, selects one subfolder based on angle preference.
    Applies overlapping segmentation.

    Args:
        bearing (str): 'DE' or 'FE'.
        fault_type (int): 0=Ball, 1=IR, 2=OR.
        diameter (str): Fault diameter (e.g., '7').
        sample_length (int): Length of each sample.
        base_path (str): Base dataset path.
        print_infor (bool): Print file import info.
        overlapping_ratio (float): Overlap for segmentation.

    Returns:
        np.array: Array of samples [num_samples, sample_length].
    """
    if bearing == 'DE':
        base_data_directory = os.path.join(base_path, '12k_Drive_End_Bearing_Fault_Data')
    else:
        base_data_directory = os.path.join(base_path, '12k_Fan_End_Bearing_Fault_Data')

    diameter_str = diameter.zfill(3)
    data_list = []
    step = max(1, int(sample_length * (1 - overlapping_ratio)))
    
    if fault_type == 2:  # Outer Race
        OR_path = os.path.join(base_data_directory, f'OR/{diameter_str}/')
        if not os.path.exists(OR_path):
            print(f"Warning: OR directory does not exist: {OR_path}")
            return np.array([])

        available_angles = os.listdir(OR_path)
        preferred_angles = ['@6', '@3', '@12'] if bearing == 'DE' else ['@3', '@6', '@12']

        chosen_subdir = None
        for angle in preferred_angles:
            for sub in available_angles:
                if angle in sub:
                    chosen_subdir = sub
                    break
            if chosen_subdir:
                break

        if not chosen_subdir:
            print(f"Warning: No angle subdir found for OR fault in {OR_path}")
            return np.array([])

        fault_dir = os.path.join(OR_path, chosen_subdir)
        if not os.path.exists(fault_dir):
            print(f"Warning: Chosen OR subdir does not exist: {fault_dir}")
            return np.array([])

        file_list = [f for f in os.listdir(fault_dir) if f.endswith('.mat') and not f.endswith('_0.mat')]

        for file in file_list:
            full_file_path = os.path.join(fault_dir, file)
            try:
                file_data = scipy.io.loadmat(full_file_path)
                for key in file_data:
                    if bearing in key:
                        raw = file_data[key].flatten()
                        for i in range(0, len(raw) - sample_length + 1, step):
                            sample = raw[i:i + sample_length]
                            data_list.append(sample)
                        if print_infor:
                            print(f' - File import: {full_file_path}')
                            print(f' - key: {key}, samples: {len(data_list)}')
                        break
            except Exception as e:
                print(f"Error reading {full_file_path}: {e}")
    else:
        fault_dict = {0: 'B/', 1: 'IR/'}
        fault_path = os.path.join(base_data_directory, fault_dict[fault_type] + diameter_str + '/')
        if not os.path.exists(fault_path):
            print(f"Warning: Path {fault_path} does not exist.")
            return np.array([])

        file_list = [f for f in os.listdir(fault_path) if f.endswith('.mat') and not f.endswith('_0.mat')]

        for file in file_list:
            full_file_path = os.path.join(fault_path, file)
            try:
                file_data = scipy.io.loadmat(full_file_path)
                for key in file_data:
                    if bearing in key:
                        raw = file_data[key].flatten()
                        for i in range(0, len(raw) - sample_length + 1, step):
                            sample = raw[i:i + sample_length]
                            data_list.append(sample)
                        if print_infor:
                            print(f' - File import: {full_file_path}')
                            print(f' - key: {key}, samples: {len(data_list)}')
                        break
            except Exception as e:
                print(f"Error reading {full_file_path}: {e}")

    if len(data_list) == 0:
        print(f'Warning: No data found for fault_type={fault_type}, diameter={diameter}, bearing={bearing}')
        return np.array([])

    return np.array(data_list)

def data_import(base_path=BASE_PATH, sample_length=SAMPLE_LENGTH, preprocessing=PREPROCESSING, overlapping_ratio=OVERLAPPING_RATIO):
    """
    Import and split CWRU bearing data into train/val/test sets based on configurable diameters.

    Ensures normal data files (e.g., _1, _2, _3) align with fault diameter splits (7, 14, 21).

    Args:
        base_path (str): Base path to CWRU dataset.
        sample_length (int): Length of each sample.
        preprocessing (bool): Whether to apply preprocessing (envelope or one-sided spectra).
        overlapping_ratio (float): Overlap ratio for data segmentation in training.

    Returns:
        tuple: (X_train, Y_train, X_val, Y_val, X_test, Y_test) as numpy arrays.
    """
    effective_sample_length = sample_length
    if PREPROCESSING_TYPE in ['envelope', 'one_sided'] and preprocessing:
        effective_sample_length *= 2
    
    fault_diameters = DIAMETER_ORDER
    print("Bearing-based split (simulating unseen machines):", fault_diameters)
    split_map = {'train': fault_diameters[0], 'val': fault_diameters[1], 'test': fault_diameters[2]}
    
    # Map suffixes to diameters (e.g., '_1' -> '7')
    suffix_to_diameter = {'_1': '7', '_2': '14', '_3': '21'}
    # Reverse split_map to map diameters to splits (e.g., '7' -> 'train')
    diameter_to_split = {v: k for k, v in split_map.items()}
    
    fault_types = [0, 1, 2]  # Ball, IR, OR
    bearings = ['DE', 'FE']
    
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []

    # Load fault data
    for split, diameter in split_map.items():
        overlap = overlapping_ratio if split == 'train' else 0.0
        for fault_type in fault_types:
            for bearing in bearings:
                data = import_file_specific(bearing, fault_type, diameter, effective_sample_length, base_path, print_infor=True, overlapping_ratio=overlap)
                if data.shape[0] == 0:
                    continue
                label = get_fault_label(fault_type, bearing)
                num_samples = data.shape[0]
                labels = [label] * num_samples

                if split == 'train':
                    X_train.append(data)
                    Y_train.extend(labels)
                elif split == 'val':
                    X_val.append(data)
                    Y_val.extend(labels)
                else:
                    X_test.append(data)
                    Y_test.extend(labels)

    # Load Normal Data (label 0)
    file_path = os.path.join(base_path, 'Normal/')
    for file in os.listdir(file_path):
        full_file_path = os.path.join(file_path, file)
        # Determine split by matching file suffix to diameter, then to split
        split = None
        for suffix, diameter in suffix_to_diameter.items():
            if suffix in file:
                split = diameter_to_split.get(diameter)
                break
        if split is None:
            continue  # Skip files not matching expected suffixes
        
        file_data = scipy.io.loadmat(full_file_path)
        overlap = overlapping_ratio if split == 'train' else 0.0
        step = max(1, int(effective_sample_length * (1 - overlap)))
        
        file_samples = []
        for key in file_data:
            if 'DE' in key or 'FE' in key:
                raw = file_data[key].flatten() 
                original_fs = 48000
                target_fs = 12000
                new_length = int(len(raw) * (target_fs / original_fs)) 
                raw = resample(raw, new_length)
                for i in range(0, len(raw) - effective_sample_length + 1, step):
                    sample = raw[i:i + effective_sample_length]
                    file_samples.append(sample)
        
        if len(file_samples) == 0:
            continue
        
        data = np.array(file_samples)
        num_samples = data.shape[0]
        labels = [0] * num_samples

        if split == 'train':
            X_train.append(data)
            Y_train.extend(labels)
        elif split == 'val':
            X_val.append(data)
            Y_val.extend(labels)
        elif split == 'test':
            X_test.append(data)
            Y_test.extend(labels)

    def finalize(X, Y):
        if len(X) == 0:
            return np.array([]), np.array([])
        X = np.vstack(X)
        Y = np.array(Y)
        return X, Y

    X_train, Y_train = finalize(X_train, Y_train)
    X_val, Y_val = finalize(X_val, Y_val)
    X_test, Y_test = finalize(X_test, Y_test)

    print('='*20, 'Training Data Distribution', '='*20)
    analyze_data(Y_train)
    print('='*20, 'Validation Data Distribution', '='*20)
    analyze_data(Y_val)
    print('='*20, 'Testing Data Distribution', '='*20)
    analyze_data(Y_test)

    if preprocessing and PREPROCESSING_TYPE != 'none':
        if PREPROCESSING_TYPE == 'envelope':
            _, X_train = batch_envelope_analysis(X_train, 12000)
            _, X_val = batch_envelope_analysis(X_val, 12000)
            _, X_test = batch_envelope_analysis(X_test, 12000)
        elif PREPROCESSING_TYPE == 'one_sided':
            _, X_train = batch_one_sided_spectra(X_train, 12000)
            _, X_val = batch_one_sided_spectra(X_val, 12000)
            _, X_test = batch_one_sided_spectra(X_test, 12000)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def batch_cepstrum_prewhitening(signals):
    """Cepstrum prewhitening for signals."""
    signals_fft = fft(signals, axis=-1)
    magnitude = np.abs(signals_fft)
    epsilon = 1e-12
    magnitude[magnitude < epsilon] = epsilon
    whitened_fft = signals_fft / magnitude
    whitened_signals_complex = ifft(whitened_fft, axis=-1)
    return np.real(whitened_signals_complex)

def batch_envelope_analysis(signals, fs):
    """Envelope analysis for time-domain signals."""
    whitened_signals = batch_cepstrum_prewhitening(signals)
    analytic_signals = hilbert(whitened_signals, axis=-1)
    envelopes = np.abs(analytic_signals)
    envelopes_mean_removed = envelopes - envelopes.mean(axis=1, keepdims=True)
    ses_ffts = fft(envelopes_mean_removed**2, axis=-1)

    N_fft = signals.shape[1]
    ses_spectra = 2.0 / N_fft * np.abs(ses_ffts[:, 0:N_fft//2])
    freq_axis = np.linspace(0.0, 0.5 * fs, N_fft//2)

    return freq_axis, ses_spectra

def batch_one_sided_spectra(signals, fs):
    """Compute one-sided amplitude spectrum."""
    num_samples, sample_length = signals.shape
    fft_vals = fft(signals, axis=-1)
    fft_magnitude = np.abs(fft_vals)
    spectra = 2.0 / sample_length * fft_magnitude[:, :sample_length // 2]
    freq_axis = np.linspace(0.0, 0.5 * fs, sample_length // 2)
    return freq_axis, spectra

def normalize_data(X_train, X_val, X_test):
    """Normalize data using Min-Max scaling based on training set."""
    print("Max min")
    train_min = np.min(X_train)
    train_max = np.max(X_train)
    if train_max - train_min == 0:
        raise ValueError("Training data has constant values; Min-Max normalization is undefined.")
    X_train_normalized = (X_train - train_min) / (train_max - train_min)
    X_val_normalized = (X_val - train_min) / (train_max - train_min)
    X_test_normalized = (X_test - train_min) / (train_max - train_min)
    return X_train_normalized, X_val_normalized, X_test_normalized