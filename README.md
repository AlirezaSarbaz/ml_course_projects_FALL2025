# machine learning practical exercises
## 1.Supervised Learning

Supervised learning is a common machine learning approach where a model is trained on labeled data to learn the relationship between input features and target outputs. The goal is to enable the model to make accurate predictions on unseen data, typically in the form of regression (predicting continuous values) or classification (predicting discrete classes).

### 1. Regression and Perceptron

This notebook contains two main parts that cover fundamental supervised learning concepts:

**Part 1: Polynomial Regression**  
- Implements polynomial regression to predict aircraft fuel consumption based on average flight speed
- Simulates a nonlinear relationship between speed and fuel consumption and creates a synthetic dataset
- Trains and compares models with different polynomial degrees (e.g., 1 to 10)
- Uses scikit-learn tools and mean squared error (MSE) for model evaluation

**Part 2: Perceptron**  
- Implements the Perceptron algorithm for binary classification
- Trains the model on linearly separable data to find a separating decision boundary
- Analyzes convergence behavior and visualizes how the model separates the two classes

### 2. Titanic Survival Prediction

This project is a classic binary classification task using the Titanic dataset and a Random Forest Classifier:

**Key features:**  
- Loads and explores the Kaggle Titanic dataset, including basic exploratory data analysis (EDA)
- Performs data cleaning and preprocessing (handling missing values, encoding categorical features, etc.)
- Uses features such as passenger class, sex, age, and fare to predict survival
- Trains a Random Forest classifier and evaluates it using metrics like accuracy, confusion matrix, and classification report

## 2.Unsupervised Learning

Unsupervised learning is a machine learning paradigm where models learn patterns and structures from unlabeled data without explicit target outputs. Unlike supervised learning, the goal is to discover hidden patterns, group similar data points, or reduce data dimensionality. Common techniques include clustering (grouping similar data) and dimensionality reduction (extracting meaningful features while reducing complexity).


### 1. K-Means Clustering

This project implements K-Means clustering from scratch and applies it to image segmentation based on color similarity:

**Project overview:**
- Implements the K-Means algorithm without using scikit-learn libraries
- Performs color-based image segmentation by clustering pixels into K distinct color groups
- Each pixel is represented by its RGB values and assigned to the nearest cluster
- Iteratively updates cluster centers until convergence

**Key features:**
- Custom implementation of K-Means initialization, assignment, and update steps
- Elbow method visualization (WCSS vs. K) to determine optimal number of clusters
- Image reshaping and pixel-level processing
- Comparison of segmentation quality with different K values

**Application:** Color-based region extraction for image processing tasks

### 2. Gaussian Mixture Model (GMM)

This project implements GMM from scratch and applies it to customer segmentation based on demographic and behavioral data :

**Project overview:**
- Implements Gaussian Mixture Model algorithm without using pre-built libraries
- Performs customer segmentation using features like Age, Annual Income, and Spending Score
- Uses probabilistic clustering where each data point has a probability of belonging to each cluster
- Applies feature standardization for better model performance

**Key features:**
- Custom GMM implementation with Expectation-Maximization (EM) algorithm
- 2D scatter plots showing cluster assignments with centroids and cluster spread visualization
- 3D visualization of customer segments across multiple features
- Semi-transparent circles representing cluster covariance and spread

**Application:** Marketing analytics and customer behavior analysis

### 3. Principal Component Analysis (PCA)

This project implements PCA from scratch and applies it to image compression by reducing dimensionality while preserving important information [file:5]:

**Project overview:**
- Implements PCA algorithm from scratch using numpy (without sklearn)
- Divides grayscale images into small patches and treats each patch as a data vector
- Reduces dimensionality by projecting patches onto principal components
- Reconstructs compressed images and evaluates reconstruction quality

**Key implementation steps:**
- Data centering by subtracting mean
- Covariance matrix computation
- Eigenvalue and eigenvector calculation
- Sorting eigenvectors by eigenvalues in descending order
- Projection and reconstruction using top n components

**Key features:**
- Patch extraction and reconstruction functions
- Visualization comparing original vs. reconstructed images
- Quality analysis showing reconstruction with reduced dimensions (e.g., 144 features to 16 components)
- Error calculation to measure compression quality

**Application:** Image compression and feature extraction

## 3.Neural Networks

Neural networks are computational models inspired by biological neural networks that constitute animal brains. They consist of interconnected nodes (neurons) organized in layers, where each connection has an associated weight that is adjusted during training. Neural networks can learn complex patterns and relationships in data through a process called backpropagation, where errors are propagated backward through the network to update weights. Multi-layer perceptrons (MLPs) are a fundamental type of neural network that can approximate any continuous function, making them powerful tools for both classification and regression tasks.

### Neural Networks & Optimization

This project implements a complete neural network framework from scratch using only NumPy, covering all essential components of deep learning including layers, activation functions, loss functions, optimizers, and training procedures:

**Project overview:**
- Implements a Multi-Layer Perceptron (MLP) from scratch without using deep learning frameworks (PyTorch/TensorFlow)
- Builds neural network layers including Linear, ReLU, Tanh, Dropout, and BatchNorm with forward and backward propagation
- Implements multiple optimization algorithms: SGD, Momentum, Nesterov, RMSProp, Adam, and AdamW
- Includes L1/L2 regularization and a complete training loop with gradient accumulation and early stopping
- Performs experiments comparing different optimizers, regularization methods, and training strategies

**Key features:**
- Complete implementation from scratch using only NumPy (no deep learning frameworks)
- Numerical gradient checking for verifying backpropagation correctness
- Modular architecture with configurable layers, activations, and regularization
- Support for micro-batching (gradient accumulation) for training with large logical batches
- Comprehensive experiments demonstrating optimizer comparison, regularization ablation, and BatchNorm impact analysis
- Training history tracking with train/validation loss and accuracy metrics

**Application:** Deep learning fundamentals, understanding neural network internals, and educational purposes for learning how neural networks work under the hood

