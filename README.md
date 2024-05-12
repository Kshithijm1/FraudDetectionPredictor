
## Fraud Detection System

This project implements a fraud detection system using an autoencoder for feature extraction and a dense neural network for classification. The dataset used in this project contains transactions from credit cards; it is specifically designed to detect fraudulent transactions.

### Prerequisites

- Python 3.6+
- pandas
- numpy
- scikit-learn
- keras
- TensorFlow
- imbalanced-learn
- joblib

You can install the necessary libraries using pip:
```bash
pip install pandas numpy scikit-learn keras tensorflow imbalanced-learn joblib
```

### Dataset

The dataset named `creditcard.csv` should be structured with transaction features labeled from `V1` to `V28`, a `Time` feature indicating the seconds elapsed between each transaction and the first transaction in the dataset, an `Amount` feature which is the transaction Amount, and a `Class` feature where 1 stands for fraudulent transaction and 0 stands for non-fraudulent.

### Feature Engineering

Advanced feature engineering steps include:
- Log transformation of the `Amount` to reduce skewness and scaling issues.
- Extracting the hour of the day from the `Time` feature, which might capture intra-day patterns.
- Interaction feature between the log-transformed amount and the hour to capture any specific patterns at different times of the day.

These features replace the original `Time` and `Amount` features in the dataset.

### Model Architecture

#### Autoencoder
- **Input Layer**: Size equal to the number of input features (after preprocessing).
- **Encoder**: Dense layers with decreasing units, regularizers, and dropout to learn compressed representation.
- **Decoder**: Mirrors the encoder architecture to reconstruct the input features.

#### Classifier
- **Input Layer**: Compressed feature set from the autoencoder.
- **Hidden Layers**: Dense layers with dropout to prevent overfitting.
- **Output Layer**: Single neuron with sigmoid activation to output the probability of fraud.

### Training

- The autoencoder and classifier are trained using Adam optimizer with early stopping based on validation loss/auc.
- The datasets are first split into training and test sets. The training data is oversampled using SMOTE to handle class imbalance.
- Features are scaled using `RobustScaler` to handle outliers effectively.

### Evaluation

- The model's performance is evaluated using ROC-AUC, precision-recall AUC, accuracy, confusion matrix, and classification report metrics.
- Predictions are made on the test set, and evaluation metrics are computed to assess the performance.

### Usage

The trained model and scaler are saved to disk. For new predictions:
1. Load the saved models and scaler.
2. Apply the same preprocessing and feature engineering steps to the new data.
3. Predict using the classifier.

### Saving and Loading Models

Models and the scaler object are saved using `model.save()` and `joblib.dump()`, respectively. This allows for easy distribution and deployment of the trained system for later use without retraining.

