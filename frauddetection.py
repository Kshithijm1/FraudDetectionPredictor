import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras import regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Advanced Feature Engineering
data['Log_Amount'] = np.log(data['Amount'] + 1)
data['Hour'] = data['Time'] % 86400 // 3600
data['Amount_Hour_Interaction'] = data['Log_Amount'] * data['Hour']
data.drop(['Time', 'Amount'], axis=1, inplace=True)

# Preparing features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalanced data with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale data
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the autoencoder architecture
input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(100, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.3)(encoded)
encoded = Dense(50, activation='relu')(encoded)
decoded = Dense(50, activation='relu')(encoded)
decoded = BatchNormalization()(decoded)
decoded = Dropout(0.3)(decoded)
decoded = Dense(100, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=5, batch_size=256, shuffle=True, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

# Extract features from the encoder part
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_1').output)
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

# Classification model on encoded features
classifier_input = Input(shape=(50,))
x = Dense(64, activation='relu')(classifier_input)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
classifier_output = Dense(1, activation='sigmoid')(x)
classifier = Model(inputs=classifier_input, outputs=classifier_output)
classifier.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
classifier.fit(X_train_encoded, y_train, epochs=5, batch_size=128, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)])

# Predictions and evaluation
y_pred = (classifier.predict(X_test_encoded) > 0.5).astype(int)
y_pred_proba = classifier.predict(X_test_encoded)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("PR AUC Score:", pr_auc)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save models and scaler
autoencoder.save('fraud_detection_autoencoder.h5')
classifier.save('fraud_detection_classifier.h5')
joblib.dump(scaler, 'scaler.pkl')

# Prediction function
def predict_fraud(new_data):
    # Load models and scaler
    autoencoder = load_model('fraud_detection_autoencoder.h5')
    classifier = load_model('fraud_detection_classifier.h5')
    scaler = joblib.load('scaler.pkl')

    # Ensure the same transformations are applied as were applied to the training data
    new_data['Log_Amount'] = np.log(new_data['Amount'] + 1)
    new_data['Hour'] = new_data['Time'] % 86400 // 3600
    new_data['Amount_Hour_Interaction'] = new_data['Log_Amount'] * new_data['Hour']

    # Ensure you only retain the features that were present during model training
    features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
                'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23',
                'V24', 'V25', 'V26', 'V27', 'V28', 'Log_Amount', 'Hour', 'Amount_Hour_Interaction']
    new_data = new_data[features]

    # Scale data
    new_data_scaled = scaler.transform(new_data)

    # Encode features
    new_data_encoded = encoder.predict(new_data_scaled)

    # Predict using the classifier
    prediction_prob = classifier.predict(new_data_encoded)
    prediction = (prediction_prob > 0.5).astype(int)  # Assuming a threshold of 0.5

    return prediction, prediction_prob

# Prepare new transaction data
new_transaction = pd.DataFrame([{
    'Time': 48000,  # Sample transaction time
    'Amount': 212.5,  # Sample transaction amount
    'V1': -3.5, 'V2': 2.8, 'V3': -5.1, 'V4': 4.2, 'V5': -3.1, 'V6': -1.2, 'V7': -2.8,
    'V8': 1.7, 'V9': -2.0, 'V10': -5.2, 'V11': 3.6, 'V12': -5.8, 'V13': 0.5, 'V14': -9.3,
    'V15': -0.1, 'V16': -4.1, 'V17': -6.2, 'V18': -2.4, 'V19': 0.5, 'V20': 0.7,
    'V21': 0.6, 'V22': 0.2, 'V23': -0.4, 'V24': -0.5, 'V25': 0.3, 'V26': 0.4,
    'V27': 0.8, 'V28': 0.3
}], index=[0])

# Run the prediction function
prediction, probability = predict_fraud(new_transaction)
print("Prediction (0=Non-Fraudulent, 1=Fraudulent):", prediction)
print("Probability of Fraud:", probability)
