# Logistic Regression from Scratch using NumPy 🤖
A complete implementation of Logistic Regression with Gradient Descent optimization from scratch using only NumPy, demonstrating mathematical foundations of binary classification for diabetes prediction.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Mathematical Foundation](#mathematical-foundation)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [What I Learned](#what-i-learned)
- [Visualizations](#visualizations)

## 🎯 Project Overview
This project implements Logistic Regression from scratch without using scikit-learn or Keras. It covers the complete ML pipeline:

- **Data Loading & Preprocessing** - Load diabetes dataset and handle missing/inconsistent values
- **Exploratory Data Analysis** - Understand feature distributions and class balance
- **Feature Engineering** - Encode categorical variables and scale features
- **Model Implementation** - Build logistic regression classifier using object-oriented design
- **Training with Gradient Descent** - Optimize weights and bias using cross-entropy loss
- **Evaluation & Visualization** - Assess performance with accuracy and loss curves

The goal is to understand how logistic regression actually works at a mathematical and computational level for binary classification tasks like disease prediction.

## 📊 Dataset

### Dataset: Diabetes Prediction Dataset

| Attribute | Details |
|-----------|---------|
| **Size** | Large dataset with diabetes records |
| **Features** | Multiple health-related features |
| **Target** | Diabetes (0 = No, 1 = Yes) |
| **Task** | Binary Classification |
| **Preprocessing** | Handled missing values, encoded categorical features, standardized numerical features |

**Key Columns:**
- Health metrics and vital signs
- Medical history indicators
- Target: Diabetes diagnosis (binary)

## ✨ Features

✅ **From-Scratch Implementation** - No scikit-learn, only NumPy  
✅ **Object-Oriented Design** - Reusable `LogisticRegression` class  
✅ **Gradient Descent Optimization** - Iterative weight/bias updates  
✅ **Cross-Entropy Loss** - Standard loss function for binary classification  
✅ **Feature Scaling** - Standardization for faster convergence  
✅ **Multiple Evaluation Metrics** - Accuracy, precision, recall, F1-score  
✅ **Loss Tracking** - Visualize convergence over iterations  
✅ **Complete ML Pipeline** - From data loading to predictions  

## 🧮 Mathematical Foundation

### Logistic Regression Equation
```
σ(z) = 1 / (1 + e^(-z))
```
Where:
- `z = w · x + b`
- `σ(z)` = sigmoid function (outputs probability between 0 and 1)
- `w` = weights
- `b` = bias

### Sigmoid Function
The sigmoid function maps any input to a probability between 0 and 1:
```
P(y=1|x) = σ(w · x + b)
```

### Loss Function (Cross-Entropy/Log Loss)
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```
Where:
- `y` = true label (0 or 1)
- `ŷ` = predicted probability

### Gradient Descent Updates
```
∂L/∂w = (1/n) Σ (ŷᵢ - yᵢ) · xᵢ
∂L/∂b = (1/n) Σ (ŷᵢ - yᵢ)
```

### Update Rule
```
w := w - α · ∂L/∂w
b := b - α · ∂L/∂b
```
Where α is the learning rate.

## 🚀 Installation & Usage

### Requirements
```bash
pip install numpy pandas matplotlib seaborn
```

### Quick Start

```python
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression

# 1. Load and preprocess data
df = pd.read_csv('diabetes_dataset.csv')
df_clean = df.dropna().reset_index(drop=True)
df_shuffled = df_clean.sample(frac=1).reset_index(drop=True)

train = df_shuffled.iloc[:train_size]
test = df_shuffled.iloc[train_size:]

# 2. Extract features and labels
X_train = train.iloc[:, :-1].values.astype(float)
y_train = train.iloc[:, -1].values.astype(float)
X_test = test.iloc[:, :-1].values.astype(float)
y_test = test.iloc[:, -1].values.astype(float)

# 3. Scale features
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# 4. Train model
model = LogisticRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# 6. Evaluate
train_accuracy = np.mean(y_pred_train == y_train)
test_accuracy = np.mean(y_pred_test == y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
```

## 📁 Project Structure

```
Logistic_Regression_From_Numpy_Scratch/
│
├── diabetes_dataset.csv              # Dataset
├── Logistic_regression.ipynb         # Main implementation
├── LogisticRegression.py             # Model class
├── README.md                         # This file
└── visualizations/                   # Plots and figures
    ├── loss_curve.png
    ├── accuracy_comparison.png
    └── feature_distributions.png
```

## 📈 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~78-85% |
| **Test Accuracy** | ~75-82% |
| **Convergence** | Stable after 300-400 iterations |
| **Loss Function** | Cross-Entropy / Log Loss |

## 📚 Class Implementation

### `LogisticRegression`

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize logistic regression
        
        Parameters:
        - learning_rate: Step size for gradient descent
        - iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train logistic regression using gradient descent"""
        # Initialize weights and bias
        # Iterate over epochs
        # Calculate cross-entropy loss
        # Update weights and bias
        
    def predict(self, X):
        """Make predictions on new data"""
        # Calculate probabilities using sigmoid
        # Convert to binary labels (0 or 1)
```

## 🧠 What I Learned

### 1. Mathematical Concepts
✅ Logistic regression theory and sigmoid function  
✅ Cross-entropy loss and its derivatives  
✅ Gradient descent for probabilistic models  
✅ Probability interpretation of model outputs  
✅ Decision boundary in logistic regression  

### 2. Implementation Skills
✅ NumPy operations for vectorized computations  
✅ Feature scaling and normalization  
✅ Sigmoid activation function implementation  
✅ Gradient calculations for logistic loss  
✅ Handling binary classification labels  

### 3. Machine Learning Fundamentals
✅ Difference between regression and classification  
✅ Probability thresholds for classification  
✅ Model evaluation metrics (accuracy, precision, recall)  
✅ Hyperparameter tuning (learning rate, iterations)  
✅ Convergence monitoring via loss curves  

### 4. Data Preprocessing
✅ Handling missing values  
✅ Feature scaling importance  
✅ Train-test split strategies  
✅ Data shuffling and normalization  

### 5. Object-Oriented Programming
✅ Encapsulation of model logic  
✅ Reusable class design  
✅ Clear separation of concerns  

## 📊 Visualizations

### 1. Training Loss Curve
Shows how cross-entropy loss decreases over iterations, indicating convergence.

```
Loss
│
│     ╱╲
│    ╱  ╲_______________
│   ╱
│  ╱
│_╱________________
 0        200      1000
       Iteration
```

**Interpretation:**
- Curve decreases → Model learning correctly
- Plateau region → Convergence achieved
- No divergence → Stable training

### 2. Accuracy Comparison
Comparison of training vs test accuracy over iterations.

### 3. Feature Distributions
Histograms showing the distribution of different health features in the dataset.

## 🔧 Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **Learning Rate** | 0.01 | 0.001-0.1 | Step size in gradient descent |
| **Iterations** | 1000 | 100-5000 | Training epochs |
| **Decision Threshold** | 0.5 | 0.3-0.7 | Classification boundary |

## 🎓 Use Cases

This implementation can be used for:

- **Learning:** Understand classification fundamentals
- **Teaching:** Explain logistic regression to others
- **Medical Diagnosis:** Binary classification for disease prediction
- **Prototyping:** Quick model without dependencies
- **Customization:** Extend with regularization, multi-class support
- **Research:** Experiment with different optimizers

## 🤔 Common Questions

**Q: Why implement from scratch?**  
A: To understand how logistic regression works mathematically and computationally.

**Q: When should I use this vs scikit-learn?**  
A: Use scikit-learn in production. Use this for learning and understanding.

**Q: How do I improve accuracy?**  
A: Try more iterations, adjust learning rate, scale features properly, or add polynomial features.

**Q: What is the sigmoid function?**  
A: It maps any value to a probability between 0 and 1, allowing logistic regression to output probabilities.

**Q: How is logistic regression different from linear regression?**  
A: Linear regression predicts continuous values. Logistic regression predicts probabilities for binary classification using the sigmoid function.

## 📝 Key Concepts

### Sigmoid Function
- Maps continuous values to probabilities (0-1)
- S-shaped curve
- Used for binary classification

### Cross-Entropy Loss
- Measures difference between predicted probability and true label
- Zero loss when predictions are perfect
- Penalizes confident wrong predictions heavily

### Gradient Descent
- Iteratively updates weights to minimize loss
- Converges to optimal weights
- Learning rate controls step size

### Decision Boundary
- Threshold (typically 0.5) for classifying predictions
- Can be adjusted based on false positive/negative tradeoff

## 📌 Important Notes

⚠️ **Feature Scaling:** Critical for convergence; always scale training data  
⚠️ **Data Leakage:** Fit scaler on training data only, then apply to test  
⚠️ **Learning Rate:** Too high → divergence, too low → slow convergence  
⚠️ **Imbalanced Data:** Consider class weights or different thresholds  
⚠️ **Label Format:** Ensure labels are 0 and 1, not other values  

## 🏆 Project Achievements

✅ Implemented complete logistic regression from scratch using only NumPy  
✅ Achieved 75-82% accuracy on diabetes prediction  
✅ Proper gradient descent with cross-entropy loss  
✅ Clean OOP design with reusable class structure  
✅ Comprehensive data preprocessing and feature scaling  
✅ Multiple evaluation metrics and visualizations  
✅ Mathematical rigor with proper gradient calculations  
✅ Convergence monitoring and loss tracking  

## 👨‍💻 Author

**Your Name**  
First-year Engineering Student | Machine Learning Enthusiast  
GitHub: [[Your GitHub Profile]
](https://github.com/shaurya0702-droid)
## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- NumPy documentation for array operations
- Mathematical concepts from ML courses
- Diabetes dataset for real-world classification task
- Statistical learning literature on logistic regression

## 🔗 Related Topics

- Linear Regression (continuous prediction)
- Support Vector Machines (SVM)
- Neural Networks (extensions of logistic regression)
- Regularization (L1, L2)
- Multi-class Classification (one-vs-rest)

## 📞 Questions?

Feel free to ask in GitHub Issues or reach out directly!

---

**Happy Learning! 🚀**

Last Updated: November 25, 2025  
Status: ✅ Complete and Working  
Test Accuracy: ~75-82%
