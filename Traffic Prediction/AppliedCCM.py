# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ConvergentCrossMapping import CCM  # Ensure the class is imported correctly
from scipy.stats import pearsonr

# Load the data
file_path = "C:/Users/kragg/OneDrive/Documents/Code/Data/uber.csv"
uber_data = pd.read_csv(file_path)

# Preprocess the data
# Convert pickup_datetime to datetime object
uber_data['pickup_datetime'] = pd.to_datetime(uber_data['pickup_datetime'])

# Remove duplicates
uber_data = uber_data.drop_duplicates()

# Scale the relevant features
scaler = StandardScaler()
uber_data[['fare_amount', 'distance_km', 'time_of_day', 'hour_of_day']] = scaler.fit_transform(
    uber_data[['fare_amount', 'distance_km', 'time_of_day', 'hour_of_day']])

# Reduce the dataset size for demonstration
uber_data_sampled = uber_data.sample(n=5000, random_state=1)  # Sample 5000 rows

# Extract the variables for CCM
X = uber_data_sampled['distance_km'].values
Y = uber_data_sampled['fare_amount'].values

# Generate indices for train, validation, and test sets
indices = np.arange(len(X))
X_train_indices, X_temp_indices, Y_train_indices, Y_temp_indices = train_test_split(
    indices, indices, test_size=0.4, random_state=1)
X_val_indices, X_test_indices, Y_val_indices, Y_test_indices = train_test_split(
    X_temp_indices, Y_temp_indices, test_size=0.5, random_state=1)

# Extract train, validation, and test sets based on indices
X_train, Y_train = X[X_train_indices], Y[X_train_indices]
X_val, Y_val = X[X_val_indices], Y[Y_val_indices]
X_test, Y_test = X[X_test_indices], Y[Y_test_indices]

# Check for overlap by comparing indices
assert len(set(X_train_indices).intersection(X_val_indices)) == 0, "Overlap detected between training and validation sets!"
assert len(set(X_train_indices).intersection(X_test_indices)) == 0, "Overlap detected between training and test sets!"
assert len(set(X_val_indices).intersection(X_test_indices)) == 0, "Overlap detected between validation and test sets!"

# Applying CCM to the training data
ccm = CCM(X_train, Y_train, tau=1, E=2, L=len(X_train))
mse_train = ccm.causality()
print(f'Training Mean Squared Error (causality measure): {mse_train}')

# Define the test_ccm function
def test_ccm(ccm, X_test, Y_test):
    predictions = []
    for t in range(len(X_test)):
        X_true, X_hat = ccm.predict(t)
        if not np.isnan(X_true) and not np.isnan(X_hat):
            predictions.append((X_true, X_hat))
    X_true_array = np.array([p[0] for p in predictions])
    X_hat_array = np.array([p[1] for p in predictions])
    mse_test = np.mean((X_true_array - X_hat_array) ** 2)
    return X_true_array, X_hat_array, mse_test

# Validate the model on the validation data
X_val_true, X_val_hat, mse_val = test_ccm(ccm, X_val, Y_val)
print(f'Validation Mean Squared Error (causality measure): {mse_val}')
pearson_corr_val, _ = pearsonr(X_val_true, X_val_hat)
print(f'Validation Pearson Correlation: {pearson_corr_val}')

# Test the model on the testing data
X_test_true, X_test_hat, mse_test = test_ccm(ccm, X_test, Y_test)
print(f'Testing Mean Squared Error (causality measure): {mse_test}')
pearson_corr_test, _ = pearsonr(X_test_true, X_test_hat)
print(f'Testing Pearson Correlation: {pearson_corr_test}')

ccm.visualize_cross_mapping()