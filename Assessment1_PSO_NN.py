# -*- coding: utf-8 -*-
"""
Created on Mon Apr 7 21:40:25 2025

@author: 12223508

Assessment 1 – Enhancing Neural Network Performance with Particle Swarm Optimization
Wine Quality Prediction with PSO-optimized Neural Networks
"""

# =============================================================================
# Section 0: Suppress TensorFlow logs and import libraries
# =============================================================================
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging messages (only errors will be shown)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Disable oneDNN optimizations for consistent performance
sys.stderr = open(os.devnull, 'w')          # Redirect standard error to null to suppress warnings

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# =============================================================================
# Section A: Data Discovery and Exploration
# =============================================================================

def explore_wine_data(red_path, white_path):
    """Explore and visualize wine data"""
    
    def read_wine_csv(path):
        # Read the header line (handles the special formatting in wine dataset)
        with open(path, 'r') as f:
            header_line = f.readline().strip()
        
        # Split and clean column names (handles ""column name"" format)
        column_names = [col.strip().strip('"') for col in header_line.split(';')]
        
        # Load data skipping original header, assigning cleaned names
        df = pd.read_csv(path, sep=';', quotechar='"', header=0, names=column_names) 
        return df
    
    # Load datasets using the helper function
    red_wine = read_wine_csv(red_path)
    white_wine = read_wine_csv(white_path)
    
    print("\n=== Data Discovery ===")
    
    # 1. Print first 5 rows of each dataset
    print("\nFirst 5 rows of Red Wine dataset:")
    print(red_wine.head())
    
    print("\nFirst 5 rows of White Wine dataset:")
    print(white_wine.head())
    
    # Add wine type as a feature
    red_wine['wine_type'] = 1      # 1 for red wine
    white_wine['wine_type'] = 0    # 0 for white wine
    
    # Combine datasets
    wine_df = pd.concat([red_wine, white_wine], ignore_index=True)
    
    # 2. Data Adjustments: Convert quality to classification (3 classes)
    quality_threshold = [0, 5, 6, 10]  # Class boundaries
    quality_labels = []
    for score in wine_df['quality']:
        if score <= quality_threshold[1]:
            quality_labels.append(0)       # Low quality (0-5)
        elif score <= quality_threshold[2]:
            quality_labels.append(1)       # Medium quality (6)
        else:
            quality_labels.append(2)       # High quality (7-10)
    
    wine_df['quality_class'] = quality_labels
    
    # 3. Class Distribution
    class_names = ['Low Quality (0-5)', 'Medium Quality (6)', 'High Quality (7-10)']
    class_counts = wine_df['quality_class'].value_counts().sort_index()
    
    print("\nClass Distribution:")
    for i, name in enumerate(class_names):
        print(f"- {name}: {class_counts[i]} samples")
    
    plt.figure(figsize=(8, 5))
    sns.countplot(x='quality_class', data=wine_df, palette='viridis')
    plt.title('Class Distribution of Wine Quality')
    plt.xlabel('Quality Class')
    plt.ylabel('Count')
    plt.xticks([0, 1, 2], class_names)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return wine_df

# =============================================================================
# Section B: Data Loading and Preprocessing
# =============================================================================

def load_wine_data(red_path, white_path):
    """Load, preprocess, and split wine data"""
    
    def read_wine_csv(path):
        # Read the header line (handles the special formatting in wine dataset)
        with open(path, 'r') as f:
            header_line = f.readline().strip()
        
        # Split and clean column names (handles ""column name"" format)
        column_names = [col.strip().strip('"') for col in header_line.split(';')]
        
        # Load data skipping original header, assigning cleaned names
        df = pd.read_csv(path, sep=';', quotechar='"', header=0, names=column_names) 
        return df

    # Load datasets using the helper function
    red_wine = read_wine_csv(red_path)
    white_wine = read_wine_csv(white_path)
    
    # Add wine type as a feature (domain knowledge enhancement)
    red_wine['wine_type'] = 1      # 1 for red wine
    white_wine['wine_type'] = 0    # 0 for white wine
    
    wine_df = pd.concat([red_wine, white_wine], ignore_index=True)
    
    # Convert quality to classification (3 classes) - problem transformation
    X = wine_df.drop('quality', axis=1)    # Features
    y = wine_df['quality']                 # Target variable
    
    # Convert continuous quality scores to discrete classes (binning)
    quality_threshold = [0, 5, 6, 10]      # Class boundaries
    quality_labels = []
    for score in y:
        if score <= quality_threshold[1]:
            quality_labels.append(0)       # Low quality
        elif score <= quality_threshold[2]:
            quality_labels.append(1)       # Medium quality
        else:
            quality_labels.append(2)       # High quality
    
    y = np.array(quality_labels)
    
    # Standardize features (mean=0, std=1) - important for neural networks
    scaler = StandardScaler()             # Feature scaling
    X_scaled = scaler.fit_transform(X)    # Transform to standardized values
    
    # Split data with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # One-hot encode target for multi-class classification (keras requirement)
    num_classes = len(np.unique(y))
    y_train_encoded = to_categorical(y_train, num_classes)
    y_test_encoded = to_categorical(y_test, num_classes)
    
    print(f"\nData loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {num_classes}")
    
    return X_train, X_test, y_train_encoded, y_test_encoded, num_classes

# =============================================================================
# Section C: Neural Network Model Building
# =============================================================================

def build_nn_model(input_dim, num_classes, hidden_layers, neurons, learning_rate, dropout_rate=0.2):
    """Build a neural network model with given hyperparameters"""
    model = Sequential()
    
    # Input layer with regularization
    model.add(Dense(neurons, activation='relu', input_dim=input_dim))   # ReLU activation for non-linearity
    model.add(Dropout(dropout_rate))                                    # Apply dropout regularization
    
    # Hidden layers (variable architecture based on hyperparameters)
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation='relu'))                    # Additional hidden layers
        model.add(Dropout(dropout_rate))                                # Regularization for each layer
    
    # Output layer (multi-class classification with softmax)
    model.add(Dense(num_classes, activation='softmax'))                 # Softmax output for class probabilities
    
    # Compile model with categorical cross-entropy (standard for classification)
    optimizer = Adam(learning_rate=learning_rate)                       # Using Adam optimizer
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# =============================================================================
# Section D: Grid Search Implementation
# =============================================================================

def grid_search(X_train, y_train, X_test, y_test, param_grid, input_dim, num_classes):
    """Traditional grid search for hyperparameter optimization (baseline approach)"""
    print("\n=== Starting Grid Search ===")
    
    best_model = None
    best_params = None
    best_accuracy = 0
    grid_history = []  # Track validation loss history
    all_models = []    # Track all models' performance
    
    # Generate all combinations of hyperparameters (cartesian product)
    combinations = []
    for hl in param_grid['hidden_layers']:
        for n in param_grid['neurons']:
            for lr in param_grid['learning_rate']:
                for dr in param_grid['dropout_rate']:
                    combinations.append({
                        'hidden_layers': hl,
                        'neurons': n,
                        'learning_rate': lr,
                        'dropout_rate': dr
                    })
    
    print(f"Total combinations to try: {len(combinations)}")
    
    # Try each combination systematically (exhaustive search)
    for i, params in enumerate(combinations):
        print(f"\nTesting combination {i+1}/{len(combinations)}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        
        # Build and train model with current hyperparameters
        model = build_nn_model(
            input_dim=input_dim, 
            num_classes=num_classes,
            hidden_layers=params['hidden_layers'],
            neurons=params['neurons'],
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate']
        )
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=20,                   # Maximum training iterations
            batch_size=16,               # Mini-batch size
            validation_split=0.2,        # Hold out 20% for validation
            callbacks=[early_stopping],  # Prevent overfitting
            verbose=0                    # Suppress output
        )
        
        # Record best validation loss
        val_loss = min(history.history['val_loss'])
        grid_history.append(val_loss)
        
        # Evaluate on test data
        _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Store model info
        all_models.append({
            'params': params.copy(),
            'accuracy': test_accuracy,
            'val_loss': val_loss
        })
        
        # Update best model if better than previous best
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model
            best_params = params
            print("  New best model found!")
    
    # Sort all models by accuracy (descending)
    top_models = sorted(all_models, key=lambda x: x['accuracy'], reverse=True)
    
    # Print best parameters (final result)
    print("\n=== Grid Search Results ===")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    
    return best_model, best_params, best_accuracy, grid_history, top_models

# =============================================================================
# Section E: PSO Implementation
# =============================================================================

def pso_optimization(X_train, y_train, X_test, y_test, param_bounds, input_dim, num_classes):
    """Optimize neural network hyperparameters using PSO"""
    print("\n=== Starting PSO Optimization ===")
    
    # PSO parameters
    swarm_size = 20             # Larger swarm size for better exploration
    max_iterations = 20         # Increase iterations to allow proper convergence
    w_start, w_end = 0.9, 0.4   # Inertia weight decay
    c1, c2 = 1.5, 1.8           # Cognitive & social coefficients (c2 higher to favor global best)
    
    # Initialize particles
    particles = []
    for _ in range(swarm_size):
        particle = {
            'position': {param: random.uniform(min_val, max_val) for param, (min_val, max_val) in param_bounds.items()},
            'velocity': {param: random.uniform(-(max_val-min_val)/4, (max_val-min_val)/4) for param, (min_val, max_val) in param_bounds.items()},
            'best_position': {},
            'best_fitness': float('inf')  # Use loss as fitness (lower is better)
        }
        particle['best_position'] = particle['position'].copy()
        particles.append(particle)
    
    # Track global best
    global_best_position = None
    global_best_fitness = float('inf')
    fitness_history = []        # Track convergence for analysis
    all_models = []             # Track all models evaluated
    
    # Function to evaluate a particle using cross-validation
    def evaluate_particle(position):
        # Convert continuous parameter values to discrete where needed
        hidden_layers = int(round(position['hidden_layers']))
        neurons = int(round(position['neurons']))
        
        model = build_nn_model(
            input_dim, num_classes, hidden_layers, neurons,
            position['learning_rate'], position['dropout_rate']
        )
        
        # K-fold cross-validation for robust evaluation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        val_losses = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            model.fit(
                X_train_fold, y_train_fold,
                epochs=30, batch_size=32,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
                verbose=0
            )
            
            val_losses.append(model.evaluate(X_val_fold, y_val_fold, verbose=0)[0])
        
        # Test accuracy
        _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Average validation loss is our fitness metric
        return np.mean(val_losses), test_accuracy, model
    
    # PSO main loop
    for iteration in range(max_iterations):
        # Calculate adaptive inertia weight
        w = w_start - iteration * ((w_start - w_end) / max_iterations)
        
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        
        # Update each particle
        for i, particle in enumerate(particles):
            # Evaluate current position
            fitness, test_accuracy, model = evaluate_particle(particle['position'])
            
            # Store model performance
            model_info = {
                'params': particle['position'].copy(),
                'accuracy': test_accuracy,
                'val_loss': fitness,
                'iteration': iteration
            }
            all_models.append(model_info)
            
            print(f"  Particle {i+1} fitness: {fitness:.6f}, accuracy: {test_accuracy:.4f}")
            
            # Update personal best
            if fitness < particle['best_fitness']:
                particle['best_fitness'] = fitness
                particle['best_position'] = particle['position'].copy()
                
                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle['position'].copy()
                    print(f"    New global best fitness: {global_best_fitness:.6f}")
            
            # Update velocity and position
            for param in param_bounds:
                r1, r2 = random.random(), random.random()  # Stochastic components
                
                # Standard PSO velocity update equation (v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x))
                cognitive = c1 * r1 * (particle['best_position'][param] - particle['position'][param])
                social = c2 * r2 * (global_best_position[param] - particle['position'][param])
                particle['velocity'][param] = w * particle['velocity'][param] + cognitive + social
                
                # Apply velocity clamping to prevent explosion
                # Limit velocity to a fraction of the search range
                max_velocity = 0.1 * (param_bounds[param][1] - param_bounds[param][0])
                particle['velocity'][param] = max(min(particle['velocity'][param], max_velocity), -max_velocity)
                
                # Update position (x = x + v)
                particle['position'][param] += particle['velocity'][param]
                
                # Enforce bounds with bounce-back strategy
                # If particle goes out of bounds, reset to boundary and dampen velocity
                min_val, max_val = param_bounds[param]
                if particle['position'][param] < min_val:
                    particle['position'][param] = min_val
                    particle['velocity'][param] *= -0.5  # Dampen velocity when hitting min boundary
                elif particle['position'][param] > max_val:
                    particle['position'][param] = max_val
                    particle['velocity'][param] *= -0.5  # Dampen velocity when hitting max boundary
        
        # Track convergence history for analysis
        fitness_history.append(global_best_fitness)
    
    # Sort models by accuracy (descending)
    top_models = sorted(all_models, key=lambda x: x['accuracy'], reverse=True)
    
    # Train final model with enhanced parameters (extended training)
    final_model = build_nn_model(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=int(round(global_best_position['hidden_layers'])),
        neurons=int(round(global_best_position['neurons'])),
        learning_rate=global_best_position['learning_rate'],
        dropout_rate=global_best_position['dropout_rate']
    )
    
    # Train with more epochs for final model (better convergence)
    final_model.fit(
        X_train, y_train,
        epochs=100,  # Extended training for final model
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    
    # Evaluate final model on test data
    _, test_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
    
    # Return best model, parameters, performance and history
    return final_model, global_best_position, test_accuracy, fitness_history, top_models

# =============================================================================
# Section F: Performance Comparison
# =============================================================================

def compare_models(grid_model, grid_params, grid_accuracy, pso_model, pso_params, pso_accuracy, X_test, y_test, grid_history, pso_history):
    """Compare traditional grid search with PSO optimization"""
    print("\n=== Model Comparison ===")
    
    # Compare accuracy metrics between approaches
    print(f"Grid Search Accuracy: {grid_accuracy:.4f}")
    print(f"PSO Accuracy: {pso_accuracy:.4f}")
    
    if pso_accuracy > grid_accuracy:
        improvement = (pso_accuracy - grid_accuracy) / grid_accuracy * 100
        print(f"PSO outperformed Grid Search by {improvement:.2f}%")  # Quantify improvement
    else:
        difference = (grid_accuracy - pso_accuracy) / grid_accuracy * 100
        print(f"Grid Search outperformed PSO by {difference:.2f}%")   # Quantify difference
    
    # Get class predictions for detailed analysis
    y_pred_grid = np.argmax(grid_model.predict(X_test), axis=1)  # Convert probabilities to class labels
    y_pred_pso = np.argmax(pso_model.predict(X_test), axis=1)    # Convert probabilities to class labels
    y_test_classes = np.argmax(y_test, axis=1)                   # Convert one-hot to class labels
    
    # Print detailed performance metrics (precision, recall, F1-score)
    print("\nGrid Search Classification Report:")
    grid_report = classification_report(y_test_classes, y_pred_grid, output_dict=True)
    print(classification_report(y_test_classes, y_pred_grid))
    
    print("\nPSO Classification Report:")
    pso_report = classification_report(y_test_classes, y_pred_pso, output_dict=True)
    print(classification_report(y_test_classes, y_pred_pso))
    
    # 1. Classification Metrics Comparison Graph
    # Create a bar graph comparing precision, recall, and F1-score for each class
    fig, axes = plt.subplots(3, 1, figsize=(6, 10))
    class_labels = ['Low (0-5)', 'Medium (6)', 'High (7-10)']
    classes = ['0', '1', '2']  # Class labels in the report dictionary
    metrics = ['precision', 'recall', 'f1-score']
    titles = ['Precision Comparison', 'Recall Comparison', 'F1-Score Comparison']
    
    for i, metric in enumerate(metrics):
        grid_values = [grid_report[c][metric] for c in classes]
        pso_values = [pso_report[c][metric] for c in classes]
        
        x = np.arange(len(classes))  # the label locations
        width = 0.35  # the width of the bars
        
        axes[i].bar(x - width/2, grid_values, width, label='Grid Search', color='darkred', alpha=0.7)
        axes[i].bar(x + width/2, pso_values, width, label='PSO', color='darkblue', alpha=0.7)
        
        # Add value labels on top of bars
        for j, v in enumerate(grid_values):
            axes[i].text(j - width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8, color='darkred')
        for j, v in enumerate(pso_values):
            axes[i].text(j + width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8, color='darkblue')
        
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(titles[i])
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(class_labels)
        axes[i].set_ylim(0, 1.0)
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # Add a single legend at the bottom middle
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.05), fontsize=10)
    
    # Add explanation text for metric interpretation
    fig.text(0.5, 0.02, 
             "Higher is better", 
             ha='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make room for the legend and text at the bottom
    plt.show()
    plt.close()
    
    # 2. Confusion Matrices with legends and annotations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    
    # Class labels with more descriptive names
    class_labels = ['Low (0-5)', 'Medium (6)', 'High (7-10)']  # Class labels for wine quality
    
    # Grid Search confusion matrix
    cm_grid = confusion_matrix(y_test_classes, y_pred_grid)
    im1 = ax1.imshow(cm_grid, cmap='Blues', interpolation='nearest')
    ax1.set_title('Grid Search Confusion Matrix', fontsize=12)
    ax1.set_xlabel('Predicted Quality', fontsize=10)
    ax1.set_ylabel('Actual Quality', fontsize=10)
    
    # Add text annotations to Grid Search confusion matrix
    thresh = cm_grid.max() / 2
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax1.text(j, i, f"{cm_grid[i, j]}", 
                    ha="center", va="center", 
                    color="white" if cm_grid[i, j] > thresh else "black",
                    fontsize=10)
    
    # Add class labels to axes
    ax1.set_xticks(np.arange(len(class_labels)))
    ax1.set_yticks(np.arange(len(class_labels)))
    ax1.set_xticklabels(class_labels)
    ax1.set_yticklabels(class_labels)
    
    # PSO confusion matrix
    cm_pso = confusion_matrix(y_test_classes, y_pred_pso)
    im2 = ax2.imshow(cm_pso, cmap='Blues', interpolation='nearest')
    ax2.set_title('PSO Confusion Matrix', fontsize=12)
    ax2.set_xlabel('Predicted Quality', fontsize=10)
    ax2.set_ylabel('Actual Quality', fontsize=10)
    
    # Add text annotations to PSO confusion matrix
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax2.text(j, i, f"{cm_pso[i, j]}", 
                    ha="center", va="center", 
                    color="white" if cm_pso[i, j] > thresh else "black",
                    fontsize=10)
    
    # Add class labels to axes
    ax2.set_xticks(np.arange(len(class_labels)))
    ax2.set_yticks(np.arange(len(class_labels)))
    ax2.set_xticklabels(class_labels)
    ax2.set_yticklabels(class_labels)
    
    # Add explanation text
    fig.text(0.5, 0.01, 
             "Low: Wine Quality scores 0-5\nMedium: Wine Quality score 6\nHigh: Wine Quality scores 7-10", 
             ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the explanation text
    plt.show()
    plt.close()
    
    # 3. PSO Best Fitness per Iteration
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pso_history) + 1), pso_history, 'b-o', linewidth=2, markersize=8)
    plt.title('PSO Best Fitness per Iteration', fontsize=10)
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Best Fitness (Validation Loss)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, len(pso_history) + 1))
    
    # Add value labels on top of points
    for i, val in enumerate(pso_history):plt.text(i + 1, val, f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Add explanation text
    plt.figtext(0.5, 0.01,"Lower is better for fitness (validation loss)", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    plt.close()
    
    # 4. Best Test Accuracy Comparison (NEW GRAPH 2)
    plt.figure(figsize=(8, 5))
    
    # Create data for the plot
    methods = ['Grid Search', 'PSO']
    accuracies = [grid_accuracy, pso_accuracy]
    
    # Create bars
    bars = plt.bar(methods, accuracies, width=0.6, color=['darkred', 'darkblue'], alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.title('Best Test Accuracy Comparison', fontsize=12)
    plt.ylabel('Accuracy', fontsize=10)
    plt.ylim(0, 1.0)  # Set y-axis limits
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add explanation text
    plt.figtext(0.5, 0.01,
                "Higher is better for accuracy",
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    plt.close()
    
    # 5. Compare PSO convergence with Grid Search evaluations
    plt.figure(figsize=(8, 5))
    
    # Plot PSO convergence
    plt.plot(pso_history, 'b-', linewidth=2, label='PSO Best Fitness')
    
    # For Grid Search, simulate progress (since it's not iterative in the same way)
    # This assumes grid_history contains validation losses from each configuration
    if grid_history:
        # Sort grid search results to show improvement over evaluations
        sorted_grid = sorted(grid_history)
        best_so_far = [sorted_grid[0]]
        for loss in sorted_grid[1:]:
            if loss < best_so_far[-1]:
                best_so_far.append(loss)
            else:
                best_so_far.append(best_so_far[-1])
        grid_iterations = np.linspace(0, len(pso_history)-1, len(best_so_far))
        plt.plot(grid_iterations, best_so_far, 'r--', linewidth=2, label='Grid Search Best Fitness')
    
    plt.title('Optimization Convergence Comparison', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Validation Loss (lower is better)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.yscale('log')
    
    # Add explanation text
    plt.figtext(0.5, 0.01, "Lower is better for validation loss", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    plt.close()
    
    # 6. Compare the actual parameter values directly    
    # Extract actual parameter values
    params = list(grid_params.keys())
    grid_actual = []
    pso_actual = []
    
    for param in params:
        # Format the values appropriately
        if param == 'hidden_layers':
            grid_actual.append(grid_params[param])
            pso_actual.append(int(round(pso_params[param])))
        elif param == 'neurons':
            grid_actual.append(grid_params[param])
            pso_actual.append(int(round(pso_params[param])))
        else:
            grid_actual.append(grid_params[param])
            pso_actual.append(pso_params[param])
    
    # Create a figure with subplots for each parameter
    fig, axs = plt.subplots(1, len(params), figsize=(12, 5))
    
    # Create a separate subplot for each parameter
    for i, param in enumerate(params):
        # Set positions for bars
        x = np.arange(1)
        width = 0.35
        
        # Create bars
        rects1 = axs[i].bar(x - width/2, [float(grid_actual[i])], width, label='Grid Search', color='darkred', alpha=0.7)
        rects2 = axs[i].bar(x + width/2, [float(pso_actual[i])], width, label='PSO', color='darkblue', alpha=0.7)
        
        # Add labels and title
        axs[i].set_title(f'{param.replace("_", " ").title()}', fontsize=8)
        axs[i].set_ylabel('Value', fontsize=8)
        axs[i].set_xticks([])
        axs[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for rect in rects1:
            height = rect.get_height()
            if height < 0.01:  # For small values like learning rate
                text = f'{height:.6f}'
            else:
                text = f'{height:.2f}'
            axs[i].annotate(text,
                          xy=(rect.get_x() + rect.get_width()/2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
                          
        for rect in rects2:
            height = rect.get_height()
            if height < 0.01:  # For small values like learning rate
                text = f'{height:.6f}'
            else:
                text = f'{height:.2f}'
            axs[i].annotate(text,
                          xy=(rect.get_x() + rect.get_width()/2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
            
    # Add a single legend at the top right of the figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    plt.suptitle('Optimal Hyperparameter Values', fontsize=10)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return {
        'grid_accuracy': grid_accuracy,
        'pso_accuracy': pso_accuracy,
        'improvement': pso_accuracy - grid_accuracy
    }

# =============================================================================
# Section G: Main Function
# =============================================================================

def main():
    """Main function to run the entire workflow"""
    print("=== Wine Quality Prediction with PSO-optimized Neural Networks ===")
    
    # 1. Explore wine data
    explore_wine_data('winequality-red.csv', 'winequality-white.csv')
    
    # 2. Load and preprocess data
    X_train, X_test, y_train, y_test, num_classes = load_wine_data(
        'winequality-red.csv', 
        'winequality-white.csv'
    )
    input_dim = X_train.shape[1]
    
    # 3. Define parameter search spaces
    grid_param_grid = {
        'hidden_layers': [2,3],                    # More options for grid search
        'neurons': [8,16],                         # More neuron options
        'learning_rate': [0.001,0.01],             # Common learning rate range
        'dropout_rate': [0.2,0.3]                  # Common dropout values
    }
    
    pso_param_bounds = {
        'hidden_layers': (1, 4),                    # Explore 1 to 4 hidden layers
        'neurons': (8, 64),                         # Explore 8 to 64 neurons per layer
        'learning_rate': (0.0001, 0.01),            # Continuous learning rate range
        'dropout_rate': (0.1, 0.5)                  # Wider regularization options
    }
    
    # 4. Traditional hyperparameter optimization with grid search
    grid_model, grid_params, grid_accuracy, grid_history, grid_top_models = grid_search(
        X_train, y_train, X_test, y_test, grid_param_grid, input_dim, num_classes
    )
    
    # 5. PSO-based hyperparameter optimization (evolutionary approach)
    pso_model, pso_params, pso_accuracy, pso_history, pso_top_models = pso_optimization(
        X_train, y_train, X_test, y_test, pso_param_bounds, input_dim, num_classes
    )
    
    # 6. Detailed performance comparison and analysis
    comparison = compare_models(
        grid_model, grid_params, grid_accuracy,
        pso_model, pso_params, pso_accuracy,
        X_test, y_test, grid_history, pso_history
    )
    
    # 7. Print top 5 models from each approach using the new function
    print_top_models(grid_top_models, title="=== Top 5 Grid Search Models ===")
    print_top_models(pso_top_models, title="=== Top 5 PSO Models ===")
    
    # 8. Discussion and conclusion
    print("\n=== Discussion and Conclusion ===\n")
    print("The Particle Swarm Optimization approach for neural network hyperparameter tuning")
    print("has been compared with traditional grid search for wine quality prediction.")
    
    if comparison['improvement'] > 0:
        print(f"\nPSO outperformed grid search by {comparison['improvement']*100:.2f}% in classification accuracy.")
        print("This demonstrates the effectiveness of PSO in finding better hyperparameters.")
    else:
        print("\nIn this case, traditional grid search performed better than PSO.")
        print("This might be due to the limited iterations or the specific search space.")
    
    print("\nAdvantages of PSO:")
    print(" 1. PSO can search a continuous parameter space rather than discrete points.")
    print(" 2. PSO requires fewer model evaluations than exhaustive grid search.")
    print(" 3. PSO balances exploration and exploitation through particle movement.")
    
    print("\nLimitations and Future Work:")
    print(" 1. PSO performance depends on its own hyperparameters (inertia, coefficients).")
    print(" 2. Future work could optimize more hyperparameters, like activation functions.")
    print(" 3. Hybrid approaches combining PSO with other methods could be explored.\n")

def print_top_models(models_list, title="Top 5 Models"):
    """Print the top N models and their parameters in a table format."""
    if not models_list:
        print("  No models to display.")
        return

    top_n = min(5, len(models_list))
    table_data = []

    print(f"\n{title}") # Print the title

    # Prepare data for the table
    for i, model_info in enumerate(models_list[:top_n]):
        rank = i + 1
        # Format accuracy
        accuracy_val = model_info.get('accuracy')
        accuracy = f"{accuracy_val:.4f}" if accuracy_val is not None else 'N/A'

        # Use discrete params if available (from PSO), else raw params
        params_dict = model_info.get('discrete_params', model_info.get('params', {}))

        # Extract and format parameters, handling potential missing values
        hl_val = params_dict.get('hidden_layers')
        n_val = params_dict.get('neurons')
        lr_val = params_dict.get('learning_rate')
        dr_val = params_dict.get('dropout_rate')

        row_data = {
            'Rank': rank,
            'Accuracy': accuracy,
            'Hidden Layers': f"{hl_val:.0f}" if isinstance(hl_val, (int, float)) else 'N/A',
            'Neurons': f"{n_val:.0f}" if isinstance(n_val, (int, float)) else 'N/A',
            'Learning Rate': f"{lr_val:.6f}" if isinstance(lr_val, float) else 'N/A',
            'Dropout Rate': f"{dr_val:.4f}" if isinstance(dr_val, float) else 'N/A'
        }
        table_data.append(row_data)

    # Create and print the DataFrame
    df = pd.DataFrame(table_data)
    # Define column order
    column_order = ['Rank', 'Accuracy', 'Hidden Layers', 'Neurons', 'Learning Rate', 'Dropout Rate']
    # Filter out columns that might be missing if params_dict was empty or inconsistent
    columns_to_display = [col for col in column_order if col in df.columns]

    print(df[columns_to_display].to_string(index=False))

if __name__ == "__main__":
    main()