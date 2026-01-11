"""
Title: Deep Learning Model for Vehicle Fuel Efficiency Prediction
Description:
Deep Neural Network using PyTorch to predict vehicle MPG 
based on technical specifications
Author: Seonwoo Kang
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define Neural Network Model
class MPGPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.08):
        super(MPGPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers with LeakyReLU (BatchNorm removed for small dataset)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU(0.1))  # Better gradient flow than ReLU
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

# 1. Data Loading and Cleaning
print("1. DATA LOADING AND CLEANING")

# Load dataset
df = pd.read_csv('auto-mpg.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values per column:")
print(df.isnull().sum())

# Check data types
print(f"\nData types:")
print(df.dtypes)

# Handle missing values in horsepower (marked as '?')
# First, replace '?' with NaN
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Fill missing horsepower values with median
horsepower_median = df['horsepower'].median()
df['horsepower'].fillna(horsepower_median, inplace=True)
print(f"\nFilled {df['horsepower'].isnull().sum()} missing horsepower values with median: {horsepower_median}")

# Basic statistics
print(f"\nBasic statistics:")
print(df.describe())

# 2. Data Preparation
print("\n2. DATA PREPARATION")

# Select features for regression (excluding car_name and origin)
feature_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']
target_column = 'mpg'

# Prepare X (features) and y (target)
X = df[feature_columns]
y = df[target_column]

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Further split training set into train and validation (80-20 of training set)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Training set size (after val split): {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeature scaling completed using StandardScaler")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)

X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
y_val_tensor = torch.FloatTensor(y_val.values).view(-1, 1).to(device)

X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1).to(device)

print(f"Data converted to PyTorch tensors")

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 3. Model Training
print("\n3. MODEL TRAINING")

# Initialize model
input_size = X_train_scaled.shape[1]
model = MPGPredictor(input_size=input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.08).to(device)

print(f"\nModel Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training configuration
num_epochs = 300
early_stopping = EarlyStopping(patience=30, min_delta=0.001)

# Learning Rate Scheduler (reduce on plateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training history
train_losses = []
val_losses = []

# Training loop
print(f"\nStarting training for {num_epochs} epochs...")
print(f"Batch size: {batch_size}")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Loss function: MSE")

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()
        val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        model.load_state_dict(early_stopping.best_model)
        break

print(f"\nModel trained successfully!")
print(f"Best validation loss: {early_stopping.best_loss:.4f}")

# 4. Model Evaluation
print("\n4. MODEL EVALUATION")

# Make predictions
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_tensor).cpu().numpy()
    y_val_pred = model(X_val_tensor).cpu().numpy()
    y_test_pred = model(X_test_tensor).cpu().numpy()

# Convert to 1D arrays
y_train_np = y_train.values
y_val_np = y_val.values
y_test_np = y_test.values
y_train_pred = y_train_pred.flatten()
y_val_pred = y_val_pred.flatten()
y_test_pred = y_test_pred.flatten()

# Calculate R-squared
r2_train = r2_score(y_train_np, y_train_pred)
r2_val = r2_score(y_val_np, y_val_pred)
r2_test = r2_score(y_test_np, y_test_pred)

# Calculate RMSE
rmse_train = np.sqrt(mean_squared_error(y_train_np, y_train_pred))
rmse_val = np.sqrt(mean_squared_error(y_val_np, y_val_pred))
rmse_test = np.sqrt(mean_squared_error(y_test_np, y_test_pred))

# Calculate MAE (Mean Absolute Error)
mae_train = mean_absolute_error(y_train_np, y_train_pred)
mae_val = mean_absolute_error(y_val_np, y_val_pred)
mae_test = mean_absolute_error(y_test_np, y_test_pred)

# Calculate MAPE (Mean Absolute Percentage Error)
mape_train = np.mean(np.abs((y_train_np - y_train_pred) / y_train_np)) * 100
mape_val = np.mean(np.abs((y_val_np - y_val_pred) / y_val_np)) * 100
mape_test = np.mean(np.abs((y_test_np - y_test_pred) / y_test_np)) * 100

# Calculate Accuracy (percentage of predictions within ±10% of actual value)
tolerance_percent = 10
accuracy_train = np.mean(np.abs((y_train_np - y_train_pred) / y_train_np) <= tolerance_percent/100) * 100
accuracy_val = np.mean(np.abs((y_val_np - y_val_pred) / y_val_np) <= tolerance_percent/100) * 100
accuracy_test = np.mean(np.abs((y_test_np - y_test_pred) / y_test_np) <= tolerance_percent/100) * 100

# Alternative: Accuracy within ±2 mpg
tolerance_mpg = 2.0
accuracy_train_mpg = np.mean(np.abs(y_train_np - y_train_pred) <= tolerance_mpg) * 100
accuracy_val_mpg = np.mean(np.abs(y_val_np - y_val_pred) <= tolerance_mpg) * 100
accuracy_test_mpg = np.mean(np.abs(y_test_np - y_test_pred) <= tolerance_mpg) * 100

print(f"\nTraining Set Performance:")
print(f"  R-squared: {r2_train:.4f} ({r2_train*100:.2f}% of variance explained)")
print(f"  RMSE: {rmse_train:.4f} mpg")
print(f"  MAE: {mae_train:.4f} mpg")
print(f"  MAPE: {mape_train:.2f}%")
print(f"  Accuracy (±{tolerance_percent}%): {accuracy_train:.2f}%")
print(f"  Accuracy (±{tolerance_mpg} mpg): {accuracy_train_mpg:.2f}%")

print(f"\nValidation Set Performance:")
print(f"  R-squared: {r2_val:.4f} ({r2_val*100:.2f}% of variance explained)")
print(f"  RMSE: {rmse_val:.4f} mpg")
print(f"  MAE: {mae_val:.4f} mpg")
print(f"  MAPE: {mape_val:.2f}%")
print(f"  Accuracy (±{tolerance_percent}%): {accuracy_val:.2f}%")
print(f"  Accuracy (±{tolerance_mpg} mpg): {accuracy_val_mpg:.2f}%")

print(f"\nTesting Set Performance:")
print(f"  R-squared: {r2_test:.4f} ({r2_test*100:.2f}% of variance explained)")
print(f"  RMSE: {rmse_test:.4f} mpg")
print(f"  MAE: {mae_test:.4f} mpg")
print(f"  MAPE: {mape_test:.2f}%")
print(f"  Accuracy (±{tolerance_percent}%): {accuracy_test:.2f}%")
print(f"  Accuracy (±{tolerance_mpg} mpg): {accuracy_test_mpg:.2f}%")

# 5. Visualization
print("\n5. VISUALIZATIONS")

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Training & Validation Loss
axes[0, 0].plot(train_losses, label='Training Loss', linewidth=2)
axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=11)
axes[0, 0].set_ylabel('Loss (MSE)', fontsize=11)
axes[0, 0].set_title('Training & Validation Loss Over Epochs', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Actual vs Predicted (Test Set)
axes[0, 1].scatter(y_test_np, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0, 1].plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual MPG', fontsize=11)
axes[0, 1].set_ylabel('Predicted MPG', fontsize=11)
axes[0, 1].set_title(f'Actual vs Predicted MPG (Test Set)\nR² = {r2_test:.4f}, RMSE = {rmse_test:.4f}', 
                     fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals Plot
residuals = y_test_np - y_test_pred
axes[0, 2].scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0, 2].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 2].set_xlabel('Predicted MPG', fontsize=11)
axes[0, 2].set_ylabel('Residuals', fontsize=11)
axes[0, 2].set_title('Residual Plot (Test Set)', fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

# 4. Performance Comparison
metrics = ['R²', 'RMSE', 'MAE']
train_metrics = [r2_train, rmse_train, mae_train]
val_metrics = [r2_val, rmse_val, mae_val]
test_metrics = [r2_test, rmse_test, mae_test]

x = np.arange(len(metrics))
width = 0.25

axes[1, 0].bar(x - width, train_metrics, width, label='Train', alpha=0.8)
axes[1, 0].bar(x, val_metrics, width, label='Validation', alpha=0.8)
axes[1, 0].bar(x + width, test_metrics, width, label='Test', alpha=0.8)
axes[1, 0].set_xlabel('Metrics', fontsize=11)
axes[1, 0].set_ylabel('Value', fontsize=11)
axes[1, 0].set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 5. Distribution of Residuals
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2, label=f'Mean: {residuals.mean():.4f}')
axes[1, 1].set_xlabel('Residuals', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Distribution of Residuals (Test Set)', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Prediction Error Distribution
errors = np.abs(y_test_np - y_test_pred)
axes[1, 2].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1, 2].axvline(x=mae_test, color='r', linestyle='--', lw=2, label=f'MAE: {mae_test:.4f}')
axes[1, 2].set_xlabel('Absolute Error (mpg)', fontsize=11)
axes[1, 2].set_ylabel('Frequency', fontsize=11)
axes[1, 2].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mpg_prediction_results.png', dpi=300, bbox_inches='tight')
print(f"\nVisualizations saved as 'mpg_prediction_results.png'")

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[feature_columns + [target_column]].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Features vs MPG', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Correlation heatmap saved as 'correlation_heatmap.png'")

# 6. Summary Report
print("\n6. SUMMARY REPORT")

print(f"""
Vehicle Fuel Efficiency Prediction Model Summary
Deep Learning Model using PyTorch

Model Architecture:
   - Input Features: {input_size}
   - Hidden Layers: [256, 128, 64] with LeakyReLU (Expanded capacity)
   - Dropout Rate: 0.08 (Minimal regularization for max performance)
   - Output: 1 (MPG prediction)
   - Total Parameters: {sum(p.numel() for p in model.parameters())}
   - Target: R² ≥ 0.90 with stable training
   
Training Configuration:
   - Optimizer: Adam (lr=0.001, with ReduceLROnPlateau scheduler)
   - LR Scheduler: Reduce by 0.5x every 10 epochs if no improvement
   - Loss Function: MSE
   - Batch Size: {batch_size}
   - Max Epochs: 300 (trained {len(train_losses)} epochs)
   - Early Stopping: Patience={early_stopping.patience}
   - Optimization: Larger network + lower dropout for R² ≥ 0.90
   
Model Performance:
   - R-squared Test: {r2_test:.4f} - The model explains {r2_test*100:.1f}% of MPG variance
   - RMSE Test: {rmse_test:.4f} mpg - Root Mean Squared Error
   - MAE Test: {mae_test:.4f} mpg - Mean Absolute Error
   - MAPE Test: {mape_test:.2f}% - Mean Absolute Percentage Error
   - Accuracy ±10%: {accuracy_test:.2f}% - Predictions within ±10% of actual
   - Accuracy ±2 mpg: {accuracy_test_mpg:.2f}% - Predictions within ±2 mpg
   
Training vs Validation vs Test Performance:
   - Training R²: {r2_train:.4f}, Validation R²: {r2_val:.4f}, Test R²: {r2_test:.4f}
   - Training RMSE: {rmse_train:.4f}, Validation RMSE: {rmse_val:.4f}, Test RMSE: {rmse_test:.4f}
   - Training MAE: {mae_train:.4f}, Validation MAE: {mae_val:.4f}, Test MAE: {mae_test:.4f}

Dataset Information:
   - Total samples: {len(df)}
   - Training samples: {len(X_train)}
   - Validation samples: {len(X_val)}
   - Testing samples: {len(X_test)}
   - Features used: {len(feature_columns)}
   - Feature names: {', '.join(feature_columns)}

Project completed successfully!
""")

plt.show()
