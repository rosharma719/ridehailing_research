import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Load the data
data = pd.read_csv('C:\\Users\\kragg\\OneDrive\\Desktop\\uber.csv')

# Preprocess the data
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['hour_of_day'] = data['pickup_datetime'].dt.hour

# Features and target
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance_km', 'time_of_day', 'hour_of_day']
target = 'fare_amount'

# Normalize the features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Prepare the dataset
def prepare_data(data, features, target, percentage=100):
    # Calculate the number of samples to use
    num_samples = int(len(data) * (percentage / 100))
    selected_data = data.iloc[:num_samples]
    X = selected_data[features].values
    y = selected_data[target].values
    return X, y

# Set percentage of data to use
percentage = 10  
X, y = prepare_data(data, features, target, percentage)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        out, _ = self.gru(x.unsqueeze(1), h0)
        out = self.fc(out[:, -1, :])
        return out

input_dim = len(features)
hidden_dim = 50
output_dim = 1
num_layers = 2
num_epochs = 10
learning_rate = 0.001

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    print(f'Average Test Loss: {total_loss / len(test_loader):.4f}')

# Instantiate and train LSTM model
lstm_model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
lstm_model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = lstm_model.to(lstm_model.device)

criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

train_model(lstm_model, train_loader, criterion, optimizer, num_epochs)
evaluate_model(lstm_model, test_loader, criterion)

# Instantiate and train GRU model
gru_model = GRUModel(input_dim, hidden_dim, output_dim, num_layers)
gru_model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gru_model = gru_model.to(gru_model.device)

criterion = nn.MSELoss()
optimizer = optim.Adam(gru_model.parameters(), lr=learning_rate)

train_model(gru_model, train_loader, criterion, optimizer, num_epochs)
evaluate_model(gru_model, test_loader, criterion)