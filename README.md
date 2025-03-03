# Credit Card Fraud Detection Using Deep Learning


## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Training](#model-training)
4. [Validation](#validation)
5. [Prediction](#prediction)
6. [Batch Prediction](#batch-prediction)
7. [Single Transaction Prediction](#single-transaction-prediction)
8. [Real-Time Fraud Detection](#real-time-fraud-detection)
9. [Contributing](#contributing)
10. [License](#license)
11. [What We Did, Why We Did It, and Results](#what-we-did-why-we-did-it-and-results)

---

## Introduction
This project involves training a deep learning model using LSTM layers to detect fraudulent credit card transactions. The model is designed to identify patterns in transaction data, such as temporal, behavioral, and geolocation features, to classify transactions as fraudulent or non-fraudulent. The goal is to assist financial institutions in real-time fraud detection and prevention.

---

## Dataset
The dataset used for this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/tusharbhadouria/credit-card-fraud-detection) from Kaggle. Below is the summary of the dataset:

### Dataset Summary
- **Total Transactions**: 1,852,394
- **Training Set**: 1,296,675 transactions (70%)
- **Testing Set**: 555,719 transactions (30%)
- **Features**: 23 (e.g., transaction amount, timestamp, geolocation, merchant details)
- **Target Variable**: `is_fraud` (binary: 0 for non-fraud, 1 for fraud)

### Preprocessing Details
- **Missing Values**: Imputed numerical columns with the mean.
- **Feature Engineering**: Extracted temporal features (hour, day of the week, season), calculated geolocation distances, and one-hot encoded categorical variables.
- **Class Imbalance**: Addressed using SMOTE (Synthetic Minority Oversampling Technique).

---

## Model Training
The model is trained using the `train_model.py` script. The training is performed on a deep learning model with the following configuration:

- **Model**: Bidirectional LSTM with Dropout and Batch Normalization
- **Input Shape**: (28 features, 1)
- **Epochs**: 50
- **Batch Size**: 256
- **Optimizer**: Nadam
- **Learning Rate**: 0.001
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### `train_model.py`
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Bidirectional
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Define the model
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(28, 1))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Nadam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(X_resampled, y_resampled, epochs=50, batch_size=256, validation_split=0.2, callbacks=[early_stopping, reduce_lr, checkpoint], class_weight={0: 1, 1: 20})
```

---

## Validation
To validate the trained model, use the `val.py` script. This script evaluates the model's performance on the test set using metrics such as AUC-ROC, F1-score, and confusion matrix.

### `val.py`
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate AUC-ROC and F1-score
auc_roc = roc_auc_score(y_test, y_pred)
print(f"AUC-ROC: {auc_roc}")
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1}")
```

---

## Prediction
The repository includes scripts for making predictions using the trained model. You can perform batch predictions, single transaction predictions, or real-time fraud detection.

### Batch Prediction
The `predict.py` script allows you to run predictions on a batch of transactions and save the results.

```python
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_model.keras')

# Load the test data
test_data = pd.read_csv('fraudTest.csv')

# Preprocess the test data
test_data = preprocess_data(test_data)

# Make predictions
predictions = model.predict(test_data.drop(columns=['is_fraud']))

# Save predictions
test_data['is_fraud_pred'] = (predictions > 0.5).astype(int)
test_data.to_csv('predictions.csv', index=False)
```

### Single Transaction Prediction
Use the `single_predict.py` script to predict whether a single transaction is fraudulent.

```python
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_model.keras')

# Define a single transaction
transaction = np.array([[feature1, feature2, ..., feature28]])

# Make prediction
prediction = model.predict(transaction)
print("Fraudulent" if prediction > 0.5 else "Non-Fraudulent")
```

### Real-Time Fraud Detection
The `realtime.py` script enables real-time fraud detection using streaming transaction data.

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('best_model.keras')

# Simulate real-time transaction data
while True:
    transaction = np.array([[feature1, feature2, ..., feature28]])  # Replace with real-time data
    prediction = model.predict(transaction)
    if prediction > 0.5:
        print("Fraudulent Transaction Detected!")
```

---

## Contributing
Contributions to this project are welcome! If you have any improvements or suggestions, feel free to create a pull request or open an issue.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## What We Did, Why We Did It, and Results

### **What We Did**
1. **Data Preprocessing**:
   - Handled missing values by imputing numerical columns with the mean.
   - Dropped unnecessary columns and one-hot encoded categorical variables.
   - Scaled the transaction amount using `StandardScaler`.

2. **Feature Engineering**:
   - Extracted temporal features such as hour, day of the week, and season from the transaction timestamp.
   - Calculated the distance between the cardholder and merchant locations using geolocation data.
   - Created new features to capture behavioral patterns.

3. **Handling Class Imbalance**:
   - Used SMOTE to oversample the minority class (fraudulent transactions) to address the severe class imbalance in the dataset.

4. **Model Building**:
   - Built a deep learning model using Bidirectional LSTM layers to capture sequential patterns in the data.
   - Added Dropout and Batch Normalization layers to prevent overfitting.
   - Compiled the model using the Nadam optimizer and binary cross-entropy loss.

5. **Model Training**:
   - Trained the model with early stopping, learning rate scheduling, and class weights to improve performance.
   - Used callbacks like `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` for better training.

6. **Model Evaluation**:
   - Evaluated the model using metrics such as AUC-ROC, F1-score, precision, recall, and confusion matrix.
   - Analyzed the precision-recall curve to understand the trade-off between precision and recall.

7. **Fraud Pattern Analysis**:
   - Analyzed fraud patterns based on business type, time, geography, transaction amount, user behavior, and merchant behavior.

### **Why We Did It**
- **Fraud Detection**: Credit card fraud is a significant problem for financial institutions and consumers. Detecting fraudulent transactions in real-time can save millions of dollars and protect consumers.
- **Class Imbalance**: The dataset is highly imbalanced, with fraudulent transactions representing a tiny fraction of the total transactions. We used SMOTE to address this issue and improve model performance.
- **Feature Engineering**: Creating new features helps the model capture complex patterns in the data, such as temporal and geolocation trends.
- **Deep Learning**: LSTM models are well-suited for sequential data, making them ideal for detecting patterns in transaction sequences.

### **Results**
- **AUC-ROC**: 0.9412
- **F1-Score**: 0.0743
- **Confusion Matrix**:
  ```
  [[501422  52152]
   [    50   2095]]
  ```
- **Fraud Pattern Insights**:
  - Fraud is most frequent in grocery stores and online shopping platforms.
  - Fraud peaks during late evening and early morning hours and is most frequent on weekends.
  - Fraud is concentrated in specific cities, particularly in Texas and Florida.
  - Fraudulent transactions tend to be higher in amount, and small transactions followed by large transactions may indicate card testing.

---
