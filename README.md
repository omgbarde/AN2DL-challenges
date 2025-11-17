# Pirate Pain Detection using Deep Learning

This project involves **pain level classification from skeletal joint motion data** using advanced neural network architectures to detect and differentiate pain levels in video sequences.

## 📌 Project Objective

The goal of this challenge is to build a deep learning model that can classify pain levels based on skeletal joint features extracted from video sequences. The task involves:

- **Data Processing**: Loading and preprocessing multivariate time-series data from skeletal joints
- **Feature Engineering**: Extracting and normalizing joint motion features
- **Multi-stage Classification**: Implementing hierarchical classification to distinguish pain levels:
  - **Stage 1**: Binary classification (no pain vs. pain)
  - **Stage 2**: Binary classification on pain samples (low pain vs. high pain)
- **Model Development**: Training multiple neural network architectures:
  - **The Normie (Baseline)**: Fully connected neural networks with temporal windowing
  - **The Layered Pain (CNN-based)**: Convolutional neural networks for spatial-temporal feature extraction
  - **Jury of Five (Ensemble)**: Multiple diverse models combined for robust predictions
- **Performance Evaluation**: Assessing accuracy, precision, recall, F1-score, and confusion matrices

---

## 📊 Dataset Overview

The **Pirate Pain Dataset** contains:
- **Training Set**: Skeletal motion sequences with pain labels (no_pain, low_pain, high_pain)
- **Validation Set**: Used for hyperparameter tuning and early stopping
- **Test Set**: Final evaluation on unseen data

### Key Features:
- **Temporal Windows**: Sequences of 50 consecutive frames with a stride of 10 frames
- **Joint Features**: Normalized skeletal joint coordinates (30+ dimensions per frame)
- **Metadata**: Sample indices, timestamps, and pain level annotations (4 pain surveys per sample)

---

## 🏗️ Project Structure

```
challenge1/
├── README.md                          # This file
├── the_normie.ipynb                   # Baseline approach (fully connected networks)
├── the_layered_pain.ipynb             # CNN-based hierarchical classification
├── jury_of_five.ipynb                 # Ensemble learning approach
├── challenge1_report.pdf              # Detailed technical report
└── dataset/
    ├── pirate_pain_train.csv          # Training data
    ├── pirate_pain_train_labels.csv   # Training labels
    └── pirate_pain_test.csv           # Test data
```

### Notebook Descriptions

#### 1. **The Normie** (`the_normie.ipynb`)
A straightforward baseline approach using fully connected neural networks for pain classification.

**Approach:**
- Load and preprocess skeletal joint data
- Apply min-max normalization on training data statistics
- Build temporal sequences using sliding windows
- Train multi-layer perceptron (MLP) networks
- Evaluate using standard metrics

**Key Outcomes:**
- Establishes baseline performance metrics
- Simple, interpretable model architecture
- Fast training and inference

---

#### 2. **The Layered Pain** (`the_layered_pain.ipynb`)
A hierarchical, CNN-based approach implementing two-stage classification for improved pain detection.

**Approach:**
- **Data Preparation**: 
  - Two-stage label mapping:
    - Stage 1: Binary classification (no_pain vs. all pain levels)
    - Stage 2: Binary classification (low_pain vs. high_pain) using only pain samples
  - Sequence construction with optimal window parameters
  
- **Architecture Design**:
  - Stage 1 Model: CNN layers for temporal feature extraction followed by classification
  - Stage 2 Model: Specialized network trained on pain samples only
  - Dropout and batch normalization for regularization
  
- **Training Strategy**:
  - Two-stage training pipeline
  - Early stopping based on validation metrics
  - Class-aware evaluation

**Key Outcomes:**
- Improved accuracy through hierarchical decomposition
- Better handling of class imbalance
- Capture of spatial-temporal patterns in motion data

---

#### 3. **Jury of Five** (`jury_of_five.ipynb`)
An ensemble learning approach combining multiple diverse architectures for robust predictions.

**Approach:**
- Train 5 different neural network models:
  - Different architectures (MLP, CNN variants, LSTM)
  - Different hyperparameters and initialization strategies
  - Different training procedures and data augmentations
  
- **Ensemble Strategy**:
  - Majority voting for final predictions
  - Confidence-based aggregation
  - Weighted averaging based on individual model performance
  
- **Robustness**:
  - Reduces model-specific biases
  - Improves generalization on unseen data
  - Provides uncertainty estimates through voting patterns

**Key Outcomes:**
- Superior robustness and generalization
- Higher confidence in predictions
- Reduced overfitting through diversity

---

## 📈 Results & Key Findings

### Model Comparison

The three approaches showcase different paradigms in deep learning:

| Approach | Complexity | Accuracy | Interpretability | Training Time |
|----------|-----------|----------|-----------------|---------------|
| **The Normie** | Low | Good | High | Fast |
| **The Layered Pain** | Medium | Excellent | Medium | Medium |
| **Jury of Five** | High | Best | Low | Slow |

### Performance Metrics

All models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

---

## 🛠️ Technologies & Libraries

- **PyTorch**: Deep learning framework
- **NumPy & Pandas**: Data manipulation and processing
- **Scikit-learn**: Machine learning utilities and metrics
- **Matplotlib & Seaborn**: Data visualization
- **TensorBoard**: Training monitoring and visualization

---

## 📌 Data Processing Pipeline

1. **Loading**: CSV files with skeletal joint coordinates and labels
2. **Normalization**: Min-max scaling using training set statistics
3. **Sequence Building**: Creating fixed-length windows from continuous time series
4. **Train-Val-Test Split**: User-stratified splitting to avoid data leakage
5. **PyTorch DataLoaders**: Batch processing with shuffling

---

## 🎯 Technical Highlights

### Two-Stage Classification Strategy (The Layered Pain)
- **Rationale**: Hierarchical decomposition reduces problem complexity
- **Implementation**: Stage 1 identifies pain presence, Stage 2 differentiates pain intensity
- **Benefit**: Leverages different information at each stage for improved accuracy

### Ensemble Voting (Jury of Five)
- **Diversity**: Multiple architectures prevent overfitting to specific model biases
- **Stability**: Majority voting provides more reliable predictions
- **Transparency**: Voting patterns reveal model confidence and disagreements

### Temporal Feature Extraction
- Sliding window approach captures motion dynamics
- Temporal context preserved through sequence length
- Stride parameter balances feature extraction density with computational efficiency

---

## 📊 Visualizations

Each notebook includes comprehensive visualizations:
- **Label Distribution**: Showing class balance across splits
- **Training Curves**: Loss and accuracy evolution during training
- **Confusion Matrices**: Detailed classification breakdown
- **ROC Curves**: Performance evaluation across classification thresholds
- **Feature Importance**: Which joints contribute most to predictions (for interpretability)

---

## 🚀 How to Run

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tensorboard
```

### Execution
1. **Baseline Model**:
   - Open `the_normie.ipynb` in Jupyter or VS Code
   - Run all cells sequentially
   - Check TensorBoard for training visualization

2. **Hierarchical Model**:
   - Open `the_layered_pain.ipynb`
   - Execute cells to train Stage 1 and Stage 2 models
   - Analyze two-stage predictions

3. **Ensemble Model**:
   - Open `jury_of_five.ipynb`
   - Train all 5 base models
   - Observe voting consensus and final predictions

---

## 📝 Key Insights

1. **Class Imbalance Handling**: Two-stage classification effectively addresses the class imbalance problem
2. **Temporal Patterns**: CNN architectures effectively capture motion patterns in skeletal data
3. **Ensemble Robustness**: Diverse models outperform single architectures, especially on edge cases
4. **Normalization Importance**: Proper feature scaling is critical for stable training
5. **Data Leakage Prevention**: User-stratified splits ensure test set independence

---

## 📜 Full Report

For comprehensive technical details including:
- Mathematical formulations
- Detailed architecture diagrams
- Complete experimental results
- Statistical analysis
- Ablation studies

See the [full report](challenge1_report.pdf) available in this directory.

---

## 👥 Authors

**Team: The Gradient Descenders**
- Lorenzo Bardelli
- Lorenzo Moretti
- Luca Zani

---

## 📚 References

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Deep Learning Best Practices for Time Series: https://arxiv.org/abs/1903.04467
- Ensemble Learning Methods: https://scikit-learn.org/stable/modules/ensemble.html

---

Feel free to explore the repository and contribute with suggestions or improvements!
