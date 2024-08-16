# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ConvergentCrossMapping import *

# Load the data
file_path = "C:/Users/kragg/OneDrive/Documents/Code/Data/uber.csv"
uber_data = pd.read_csv(file_path)

# Preprocess the data
uber_data['pickup_datetime'] = pd.to_datetime(uber_data['pickup_datetime'])  # Convert pickup_datetime to datetime object
uber_data = uber_data.drop_duplicates()  # Remove duplicates

# Scale the relevant features
scaler = StandardScaler()
uber_data[['fare_amount', 'distance_km', 'time_of_day', 'hour_of_day']] = scaler.fit_transform(
    uber_data[['fare_amount', 'distance_km', 'time_of_day', 'hour_of_day']]
)

# Reduce the dataset size for demonstration
uber_data_sampled = uber_data.sample(n=5000, random_state=1)

# Extract the variables for CCM and nearest neighbors
X = uber_data_sampled[['distance_km', 'time_of_day', 'hour_of_day']].values  # Exclude 'fare_amount' from X
Y = uber_data_sampled['fare_amount'].values  # Use 'fare_amount' as Y

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

# Applying CCM to the training data and testing on unseen data
ccm = CCM(X_train, Y_train, tau=1, E=4, L=len(X_train))  # Using E=4 based on multi-dimensional factors
correlation_test = ccm.calculate_correlation(X_test, Y_test)
print(f'Testing Pearson Correlation (causality measure): {correlation_test}')

# Apply nearest neighbors algorithm
nn_model = NearestNeighbors(X_train, Y_train, n_neighbors=5)
Y_pred_test = nn_model.predict(X_test)

# Calculate Mean Squared Error for Nearest Neighbors predictions
mse_nn_test = np.mean((Y_pred_test - Y_test) ** 2)
print(f'Testing Mean Squared Error (Nearest Neighbors): {mse_nn_test}')
