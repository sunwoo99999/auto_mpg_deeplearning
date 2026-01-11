# Auto MPG Prediction Project - Deep Learning Edition

## Description

A deep learning model that predicts vehicle fuel efficiency (MPG) based on automobile technical specifications using PyTorch. This project implements a neural network with multiple fully connected layers to predict miles per gallon using features such as weight, model year, acceleration, displacement, cylinders, and horsepower.

The model achieves an R-squared score of **≥ 0.90**, meaning it explains over 90% of the variance in MPG values with an average prediction error of approximately 1.7-1.8 MPG.

## Installation

Install the required dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
```

Run the prediction script:

```bash
python mpg_prediction.py
```

## Tech Stack

- Python 3.x
- **PyTorch** for deep learning framework
- pandas for data processing and manipulation
- numpy for numerical computations
- matplotlib for data visualization
- seaborn for statistical data visualization
- scikit-learn for data preprocessing and metrics

## Features

### Data Processing

- Handles missing values in horsepower column using median imputation
- Excludes non-numeric car name column from analysis
- Applies StandardScaler for feature normalization
- Converts data to PyTorch tensors for neural network training
- Three-way split: Train (64%) / Validation (16%) / Test (20%)

### Model Architecture

- **Input Layer**: 6 features (cylinders, displacement, horsepower, weight, acceleration, model year)
- **Hidden Layers**: [256, 128, 64] neurons with decreasing capacity
- **Activation Function**: LeakyReLU(0.1) for better gradient flow
- **Regularization**: Dropout (0.08) to prevent overfitting
- **Output Layer**: Single neuron for MPG prediction
- **Total Parameters**: ~35,000

### Training Configuration

- **Optimizer**: Adam with learning rate 0.001
- **Learning Rate Scheduler**: ReduceLROnPlateau (reduces LR by 0.5x when validation loss plateaus)
- **Loss Function**: MSE (Mean Squared Error)
- **Batch Size**: 32
- **Max Epochs**: 300
- **Early Stopping**: Patience of 30 epochs to prevent overfitting
- **GPU Support**: Automatically uses CUDA if available

### Model Performance

- **R-squared Test Set**: ≥ 0.90 (90%+)
- **RMSE Test Set**: ~2.2-2.3 mpg
- **MAE Test Set**: ~1.7-1.8 mpg
- **MAPE**: ~7-8%
- **Accuracy (±10%)**: ~65-70%
- **Accuracy (±2 mpg)**: ~60-65%

### Key Improvements vs Linear Regression

| Metric | Linear Regression | Deep Learning | Improvement |
| ------ | ----------------- | ------------- | ----------- |
| R²     | 0.8244            | ≥ 0.90        | +9%         |
| RMSE   | 3.07 mpg          | ~2.2 mpg      | -28%        |
| MAE    | -                 | ~1.7 mpg      | New metric  |

### Dataset Information

- Total Samples: 398
- Training Data: 254 samples (64%)
- Validation Data: 64 samples (16%)
- Test Data: 80 samples (20%)
- Features Used: 6
- Target Variable: mpg

### Generated Outputs

1. **mpg_prediction.py** - Main deep learning script with PyTorch implementation
2. **mpg_prediction_results.png** - Comprehensive visualizations including:
   - Training & validation loss curves
   - Actual vs predicted MPG scatter plot
   - Residual plot
   - Performance comparison (Train/Val/Test)
   - Residual distribution
   - Prediction error distribution
3. **correlation_heatmap.png** - Feature correlation matrix

### Advanced Features

- **LeakyReLU Activation**: Prevents dying neuron problem
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Early Stopping**: Automatic training termination when no improvement
- **Batch Training**: Mini-batch gradient descent for efficiency
- **Validation Monitoring**: Prevents overfitting with separate validation set
- **GPU Acceleration**: CUDA support for faster training

## Model Evolution

### Version 1: Linear Regression (Baseline)

- Simple scikit-learn LinearRegression
- R² = 0.8244, RMSE = 3.07 mpg

### Version 2: Initial Deep Learning

- Hidden layers: [64, 32, 16]
- R² = 0.8641, RMSE = 2.70 mpg

### Version 3: Optimized Architecture

- Hidden layers: [128, 64, 32]
- Dropout: 0.1, LR: 0.0005
- R² = 0.9022, RMSE = 2.29 mpg

### Version 4: Final Optimization (Current)

- Hidden layers: **[256, 128, 64]**
- Dropout: **0.08**, LR: **0.001**
- LeakyReLU + LR Scheduler + Early Stopping
- **R² ≥ 0.90**, **RMSE ~2.2 mpg**

## Usage Example

```python
# The script automatically:
# 1. Loads and preprocesses data
# 2. Trains the neural network
# 3. Evaluates on test set
# 4. Generates visualizations
# 5. Prints comprehensive performance report

python mpg_prediction.py
```

## Reference

[Predictive Modeling of Auto MPG and Analysis of Engine Design Factors: Comparing the Impact of Weight and Displacement](https://zenodo.org/records/18203989?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjMyZWEzYzhiLTljMmYtNGZjMi1hNjRkLTIxYjQ4M2E4NzBmYSIsImRhdGEiOnt9LCJyYW5kb20iOiIzYjkwODc0OGY4N2Y3ZWFhYWExNGIxYzExZjc3NGJlNSJ9.waGFIea8cgpsJOriyG2i3lunsxJq8IHHMjn31PpENKQ2Fe9UTekma_b5Rgc8eJus3V3lqU624NapFfOh5FK81Q)

## License

MIT License

## Author

Seonwoo Kang
Date: 2026-01-11
