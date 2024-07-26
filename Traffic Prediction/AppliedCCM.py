# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ConvergentCrossMapping import CCM  # Ensure the class is imported correctly

# Load the data
file_path = "C:/Users/kragg/OneDrive/Documents/Code/Data/uber.csv"
uber_data = pd.read_csv(file_path)

# Preprocess the data
# Convert pickup_datetime to datetime object
uber_data['pickup_datetime'] = pd.to_datetime(uber_data['pickup_datetime'])

# Scale the relevant features
scaler = StandardScaler()
uber_data[['fare_amount', 'distance_km', 'time_of_day', 'hour_of_day']] = scaler.fit_transform(
    uber_data[['fare_amount', 'distance_km', 'time_of_day', 'hour_of_day']])

# Reduce the dataset size for demonstration
uber_data_sampled = uber_data.sample(n=5000, random_state=1)  # Sample 5000 rows

# Extract the variables for CCM
X = uber_data_sampled['distance_km'].values
Y = uber_data_sampled['fare_amount'].values

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Applying CCM to the training data
ccm = CCM(X_train, Y_train, tau=1, E=2, L=len(X_train))
mse_train = ccm.causality()
print(f'Training Mean Squared Error (causality measure): {mse_train}')

# Test the model on the testing data
def test_ccm(ccm, X_test, Y_test):
    predictions = []
    for t in range(len(X_test)):
        X_true, X_hat = ccm.predict(t)
        if not np.isnan(X_true) and not np.isnan(X_hat):
            predictions.append((X_true, X_hat))
    X_true_array = np.array([p[0] for p in predictions])
    X_hat_array = np.array([p[1] for p in predictions])
    mse_test = np.mean((X_true_array - X_hat_array) ** 2)
    return mse_test

mse_test = test_ccm(ccm, X_test, Y_test)
print(f'Testing Mean Squared Error (causality measure): {mse_test}')

ccm.visualize_cross_mapping()
