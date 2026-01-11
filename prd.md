## PRD: Fuel Efficiency (mpg) Prediction Model

### 1. Overview

- Project Name: Deep Learning Model for Vehicle Fuel Efficiency Prediction.
- Objective: To build a neural network model that predicts a vehicle's fuel efficiency (mpg) based on its technical specifications using PyTorch.
- Target User: Automotive engineers, data analysts, or consumers looking for car efficiency insights.

### 2. Data Strategy

- Dataset: UCI Machine Learning Repository - Auto MPG dataset.
- Input Features (X):
- cylinders, displacement, horsepower, weight, acceleration, model_year.

- Target Variable (y):
- mpg (Miles Per Gallon).

### 3. Functional Requirements

- Data Cleaning:
- Identify and handle missing values (especially in the horsepower column).
- Filter out non-numeric columns (like car_name) for the regression calculation.

- Data Processing:
- Perform Feature Scaling (Standardization) to ensure all variables (like weight vs. cylinders) are on the same scale.
- Convert data to PyTorch tensors for neural network training.

- Model Training:
- Build a simple Deep Neural Network using PyTorch.
- Architecture: Multiple fully connected layers with ReLU activation and Dropout for regularization.
- Split the data into Training (80%) and Testing (20%) sets.
- Train using Adam optimizer and MSE loss function.
- Implement early stopping to prevent overfitting.

- Evaluation:
- Calculate R-squared to measure explanatory power.
- Calculate RMSE (Root Mean Squared Error) to measure prediction accuracy.
- Calculate MAE (Mean Absolute Error) for additional accuracy measurement.
- Track training and validation loss over epochs.

### 4. Technical Stack

- Language: Python 3.x
- Libraries:
- Pandas (Data manipulation)
- Seaborn and Matplotlib (Visualization)
- PyTorch (Deep Learning framework)
- NumPy (Numerical computations)

---

### Model Architecture

- Neural Network Design:
- Input Layer: 6 features (cylinders, displacement, horsepower, weight, acceleration, model_year)
- Hidden Layers: 2-3 fully connected layers with decreasing neuron counts
- Activation Function: ReLU (Rectified Linear Unit)
- Regularization: Dropout layers to prevent overfitting
- Output Layer: Single neuron for mpg prediction

- Training Configuration:
- Optimizer: Adam (Adaptive Moment Estimation)
- Loss Function: MSE (Mean Squared Error)
- Batch Size: 32-64
- Epochs: 100-200 with early stopping
- Learning Rate: 0.001-0.01

---

### Metric Definition

- R-squared: This represents the scorecard of the model. It shows how well the model explains the changes in fuel efficiency.
- RMSE: This represents the margin of error. It is the average difference between the predicted mpg and the actual mpg.
- MAE: Mean Absolute Error shows the average magnitude of prediction errors.
- Training Loss: Tracks how well the model is learning during training.
- Validation Loss: Monitors overfitting by tracking performance on unseen data.
