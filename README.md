# 🎙️ Brazilian Accent Detector - Deep Learning Project

A comprehensive deep learning system for detecting and classifying Brazilian Portuguese accents using audio analysis and neural networks.

## 📋 Project Overview

This project implements multiple deep learning architectures to classify Brazilian Portuguese accents from different states. It demonstrates key concepts in audio processing, neural network design, and machine learning best practices suitable for academic research and presentation.

### Key Features

- **Multiple Neural Network Architectures**: CNN, ResNet, Attention-based CNN, and LSTM
- **Audio Feature Extraction**: Mel-spectrograms, MFCC, Chroma, and Spectral Contrast
- **Data Augmentation**: Time stretching, pitch shifting, and noise injection
- **Training Optimizations**: Mixed precision training, multi-threaded data loading, early stopping
- **Comprehensive Visualizations**: Training metrics, confusion matrices, and data analysis

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended
- GPU with 4GB+ VRAM (optional but recommended)

## 📦 Installation

### 1. Clone the Repository
```bash
cd detector_sotaque
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
detector_sotaque/
├── sotaque-brasileiro-data/          # Dataset
│   ├── accent/                       # Audio files (.wav)
│   └── sotaque-brasileiro.csv        # Metadata
├── dataset.py                        # Custom Dataset and DataLoaders
├── models.py                         # Neural Network Architectures
├── train.py                          # Training Script
├── predict.py                        # Inference Script
├── analyze_data.py                   # Exploratory Data Analysis
├── main.py                           # CLI Interface
├── config.py                         # Configuration Settings
├── utils.py                          # Utility Functions
├── requirements.txt                  # Dependencies
├── data_analysis/                    # EDA Visualizations (generated)
├── experiments/                      # Model Checkpoints & Results (generated)
└── README.md                         # Documentation
```

## 🚀 Usage

### 1. Exploratory Data Analysis

Analyze the dataset distribution and characteristics:

```bash
python main.py analyze
```

This generates:
- Dataset distribution visualizations
- Statistical summaries
- State migration patterns
- Detailed JSON report

### 2. Training

#### Basic Training
```bash
python main.py train
```

#### Custom Configuration
```bash
python main.py train --model attention_cnn --epochs 50 --batch-size 16 --lr 0.001
```

#### Available Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | attention_cnn | Model architecture (cnn, resnet, attention_cnn, lstm) |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 16 | Batch size |
| `--workers` | 4 | Number of parallel data loading workers |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 15 | Early stopping patience |
| `--label-column` | birth_state | Label column (birth_state or current_state) |

### 3. Prediction

After training, use the model for inference:

```bash
# Single audio file
python main.py predict --checkpoint experiments/[EXPERIMENT]/best_model.pth --input audio.wav

# Directory of audio files
python main.py predict --checkpoint experiments/[EXPERIMENT]/best_model.pth --input audios/ --output results.json
```

### 4. List Trained Models
```bash
python main.py list-experiments
```

### 5. Interactive Mode
```bash
python main.py interactive
```

## 🧠 Neural Network Architectures

### 1. CNN (Convolutional Neural Network)
- 4 convolutional blocks with batch normalization
- MaxPooling layers for spatial downsampling
- Fast training and inference
- Best for: Baseline experiments and resource-constrained environments

### 2. ResNet (Residual Network)
- Deep architecture with skip connections
- Addresses vanishing gradient problem
- Better generalization capabilities
- Best for: High-accuracy requirements with sufficient data

### 3. Attention CNN (Recommended)
- CNN with self-attention mechanisms
- Focuses on discriminative audio features
- Spatial and channel attention modules
- Best for: Overall performance and interpretability

### 4. LSTM (Long Short-Term Memory)
- Recurrent architecture for temporal sequences
- Captures long-range dependencies
- Slower training but effective for temporal patterns
- Best for: Temporal feature modeling

## 📊 Audio Feature Extraction Pipeline

```
Audio WAV File (16kHz)
    ↓
Normalization & Preprocessing
    ↓
Feature Extraction
    ├── Mel-Spectrogram (128 bands)
    ├── MFCC (13 coefficients + Δ + ΔΔ)
    ├── Chroma Features (12 bins)
    └── Spectral Contrast (7 bands)
    ↓
Data Augmentation (Training Only)
    ├── Time Stretching (±10%)
    ├── Pitch Shifting (±2 semitones)
    └── Gaussian Noise
    ↓
Neural Network
    ↓
Classification (State Labels)
```

## ⚡ Optimization Techniques

### 1. Multi-threaded Data Loading
```python
DataLoader(
    dataset,
    num_workers=4,          # Parallel workers
    pin_memory=True,        # GPU optimization
    persistent_workers=True # Keep workers alive
)
```

### 2. Mixed Precision Training
- Uses float16 for memory efficiency
- Maintains float32 where necessary for stability
- ~2x speedup on modern GPUs

### 3. Learning Rate Scheduling
- ReduceLROnPlateau scheduler
- Automatically reduces LR when validation plateaus
- Helps fine-tune convergence

### 4. Early Stopping
- Monitors validation F1-score
- Prevents overfitting
- Saves training time

## 📈 Results & Evaluation

The system automatically generates:

### During Training
- Loss and accuracy curves (train/validation)
- F1-Score, Precision, Recall per epoch
- Best model checkpointing

### After Training
- **Training History Graph**: `training_history.png`
- **Confusion Matrix**: `confusion_matrix_teste.png`
- **Classification Report**: Detailed metrics per class
- **Training Info**: `training_info.json` with hyperparameters and results
- **Evaluation Metrics**: `evaluation_teste.json` with test set performance

### Example Output Structure
```
experiments/
└── attention_cnn_20241201_120000/
    ├── best_model.pth                    # Trained model
    ├── training_history.png              # Loss/accuracy plots
    ├── confusion_matrix_teste.png        # Test set confusion matrix
    ├── training_info.json                # Training configuration
    └── evaluation_teste.json             # Test metrics
```

## 🎓 Machine Learning Best Practices Implemented

- ✅ **Stratified Train/Val/Test Split**: Maintains class distribution across splits
- ✅ **Reproducibility**: Fixed random seeds across libraries
- ✅ **Data Augmentation**: Only applied to training set
- ✅ **Regularization**: Dropout, batch normalization
- ✅ **Comprehensive Logging**: All metrics tracked and saved
- ✅ **Checkpoint Management**: Automatic best model saving
- ✅ **Cross-validation Ready**: Easy adaptation for k-fold CV
- ✅ **Performance Monitoring**: Real-time metrics during training

## 💡 Performance Tips

### Improving Accuracy
1. Increase number of epochs (100+)
2. Use Attention CNN or ResNet architecture
3. Tune learning rate (try 0.0001 or 0.0005)
4. Adjust data augmentation parameters
5. Experiment with different feature combinations

### Faster Training
1. Use GPU with CUDA support
2. Increase batch size (if memory permits)
3. Increase num_workers (4-8 typically optimal)
4. Enable mixed precision training
5. Use CNN instead of deeper architectures

### Memory Optimization
1. Reduce batch size
2. Use lighter model (CNN)
3. Reduce number of mel bands
4. Disable mixed precision if causing issues

## 🐛 Troubleshooting

### Out of Memory Error
```bash
# Solution: Reduce batch size
python main.py train --batch-size 8
```

### CUDA Out of Memory
```bash
# Option 1: Reduce batch size
python main.py train --batch-size 4

# Option 2: Use CPU
python main.py train --device cpu
```

### DataLoader Workers Error (Windows)
```bash
# Solution: Set workers to 0
python main.py train --workers 0
```

### Model Not Learning (Loss Plateau)
```bash
# Solution: Reduce learning rate
python main.py train --lr 0.0001
```

## 📊 Dataset Information

The dataset contains:
- **819 audio samples** of Brazilian Portuguese speech
- **11 state labels**: BA, CE, DF, ES, MG, PE, PR, RJ, RN, RS, SP
- **Metadata**: Birth state, current state, age, gender, profession, text

Classification targets:
- Birth state (`--label-column birth_state`)
- Current state (`--label-column current_state`)

## 🔬 Academic Context

This project demonstrates key concepts in:
- **Deep Learning**: Multiple architectures and training strategies
- **Audio Signal Processing**: Feature extraction from raw waveforms
- **Data Science**: EDA, visualization, statistical analysis
- **Software Engineering**: Modular design, CLI interfaces, logging
- **Research Methodology**: Reproducibility, evaluation metrics, ablation studies

Ideal for:
- AI/ML course projects and presentations
- Research in speech recognition and accent classification
- Learning practical deep learning implementation
- Understanding audio processing pipelines

## 📝 Citation

If you use this project in your research or presentation, please cite:

```
Brazilian Accent Detector - Deep Learning Project
https://github.com/[your-username]/detector_sotaque
```

## 📚 References

- PyTorch Documentation: https://pytorch.org/docs/
- Librosa (Audio Processing): https://librosa.org/
- Deep Learning for Audio: https://arxiv.org/abs/1912.10544

## 📧 Contact

For questions, issues, or contributions, please open an issue in the repository.

---

**Built with PyTorch and best practices in Deep Learning**
