import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from IPython.display import display, clear_output

# Function to generate a connected grid adjacency matrix
def generate_connected_grid_adjacency_matrix(num_nodes, skip_prob=0.15, extra_edges=0.15):
    if num_nodes <= 1:
        raise ValueError("Number of nodes must be greater than 1")

    grid_size = int(np.ceil(np.sqrt(num_nodes)))
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    def add_edge(i, j):
        adjacency_matrix[i][j] = 1
        adjacency_matrix[j][i] = 1

    # Create a grid with potential missing connections
    for i in range(num_nodes):
        if i % grid_size != grid_size - 1 and i + 1 < num_nodes:  # Connect to right neighbor
            if random.random() > skip_prob:
                add_edge(i, i + 1)
        if i + grid_size < num_nodes:  # Connect to bottom neighbor
            if random.random() > skip_prob:
                add_edge(i, i + grid_size)

    # Ensure the graph is connected
    G = nx.from_numpy_array(adjacency_matrix)
    while not nx.is_connected(G):
        # Find disconnected components
        components = list(nx.connected_components(G))
        if len(components) > 1:
            # Connect a node from the largest component to a node from another component
            largest_component = max(components, key=len)
            for component in components:
                if component != largest_component:
                    node1 = random.choice(list(largest_component))
                    node2 = random.choice(list(component))
                    add_edge(node1, node2)
                    G = nx.from_numpy_array(adjacency_matrix)
                    break

    # Add extra random edges
    num_extra_edges = int(extra_edges * num_nodes)
    edges_added = 0
    while edges_added < num_extra_edges:
        node1 = random.randint(0, num_nodes - 1)
        node2 = random.randint(0, num_nodes - 1)
        if node1 != node2 and adjacency_matrix[node1][node2] == 0:
            add_edge(node1, node2)
            edges_added += 1

    return adjacency_matrix

# Utility function to generate a complex seasonal pattern
def generate_complex_seasonal_pattern(length, base_amplitude=0.5, noise_level=0.1, num_waves=3, periods=20):
    t = np.arange(length)
    seasonal_pattern = np.zeros(length)
    period_length = length // periods

    for _ in range(num_waves):
        frequency = np.random.uniform(0.5, 2.0)
        amplitude = base_amplitude * np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, 2 * np.pi)
        for p in range(periods):
            start = p * period_length
            end = start + period_length
            seasonal_pattern[start:end] += amplitude * np.sin(frequency * 2 * np.pi * t[start:end] / period_length + phase)
            
    seasonal_pattern += noise_level * np.random.normal(size=length)
    return 1 + seasonal_pattern  # Ensure the factor is always positive

# Function to generate the taxi pickup time series
def generate_pickup_time_series(num_nodes, length, mean, std_dev, exogenous_pct, skip_prob=0.15, extra_edges=0.15, periods=10):
    # Generate the adjacency matrix
    adj_matrix = generate_connected_grid_adjacency_matrix(num_nodes, skip_prob, extra_edges)

    # Initialize the pickup matrix
    pickups = np.zeros((length, num_nodes), dtype=float)
    
    # Generate the complex seasonal patterns for each node
    seasonal_patterns = np.array([generate_complex_seasonal_pattern(length) for _ in range(num_nodes)])
    
    # Generate initial pickups with complex seasonal patterns
    for t in range(length):
        for i in range(num_nodes):
            pickups[t, i] = np.random.normal(mean * seasonal_patterns[i, t], std_dev)
            if pickups[t, i] < 0:
                pickups[t, i] = 0

    # Generate exogenous influence
    G = nx.from_numpy_array(adj_matrix)
    distance_matrix = dict(nx.all_pairs_shortest_path_length(G))
    
    for t in range(1, length):
        new_pickups = np.zeros(num_nodes)
        for i in range(num_nodes):
            exogenous_sum = 0
            endogenous_pickups = pickups[t, i]
            for j in range(num_nodes):
                if i != j and j in distance_matrix[i]:
                    distance = distance_matrix[i][j]
                    weight = 1 / (1 + distance)  # Discount based on distance
                    exogenous_sum += pickups[t-1, j] * weight
            
            exogenous_pickups = exogenous_sum / (num_nodes - 1)
            new_pickups[i] = (1 - exogenous_pct) * endogenous_pickups + exogenous_pct * exogenous_pickups
            
            if new_pickups[i] < 0:
                new_pickups[i] = 0
        
        pickups[t] = new_pickups
    
    return pickups, adj_matrix

# Define the GraphConvLayer
class GraphConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, adj_matrix):
        super(GraphConvLayer, self).__init__()
        self.adj_matrix = adj_matrix
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x):
        support = torch.matmul(x, self.weight)
        out = torch.matmul(self.adj_matrix, support)
        return out

# Define the TemporalConvLayer
class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size-1) * dilation)
    
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.conv.padding[0]]  # Remove padding

# Define the GraphWaveNet
class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, adj_matrix, num_layers, kernel_size=2, dropout_rate=0.3):
        super(GraphWaveNet, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList([GraphConvLayer(input_dim, input_dim, adj_matrix) for _ in range(num_layers)])
        self.tcn_layers = nn.ModuleList([TemporalConvLayer(input_dim, input_dim, kernel_size, 2 ** i) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        for i in range(self.num_layers):
            print(f'Layer {i} input shape: {x.shape}')
            x = self.gcn_layers[i](x)
            print(f'Layer {i} after GCN shape: {x.shape}')
            x = self.tcn_layers[i](x.transpose(1, 2)).transpose(1, 2)
            print(f'Layer {i} after TCN shape: {x.shape}')
            x = torch.relu(x)
            x = self.dropout(x)
        out = self.output_layer(x)
        print(f'Output shape: {out.shape}')
        return out

# Define parameters
num_nodes = 25  # For a 5x5 grid
length = 500
mean = 100
std_dev = 10
exogenous_pct = 0.5

# Generate pickup series and adjacency matrix
pickup_series, adj_matrix = generate_pickup_time_series(num_nodes, length, mean, std_dev, exogenous_pct)

# Standardize the pickup series
scaler = StandardScaler()
pickup_series_scaled = scaler.fit_transform(pickup_series)

# Convert adjacency matrix to tensor
adj_matrix = torch.FloatTensor(adj_matrix)

# Split data into training and testing sets
train_size = length // 2
train_data = pickup_series_scaled[:train_size]
test_data = pickup_series_scaled[train_size:]

# Convert to PyTorch tensors
train_tensor = torch.FloatTensor(train_data).unsqueeze(-1)  # Shape: (train_size, num_nodes, 1)
test_tensor = torch.FloatTensor(test_data).unsqueeze(-1)  # Shape: (test_size, num_nodes, 1)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(train_tensor, train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
input_dim = train_tensor.shape[-1]
output_dim = train_tensor.shape[-1]
num_layers = 3  # Simplify the model initially
kernel_size = 2
dropout_rate = 0.3

model = GraphWaveNet(num_nodes, input_dim, output_dim, adj_matrix, num_layers, kernel_size, dropout_rate)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Experiment with learning rates

# Training loop with integrated visualization
train_losses = []
test_losses = []
num_epochs = 10  # Use fewer epochs initially for debugging
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Logging the average loss per epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")

    # Evaluation and plotting after each epoch
    model.eval()
    with torch.no_grad():
        test_output = model(test_tensor)
        test_loss = criterion(test_output, test_tensor)
        test_losses.append(test_loss.item())

    # Update plots
    ax1.clear()
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Test Loss')

    # Transforming the outputs back to the original scale for visualization
    test_output_transformed = scaler.inverse_transform(test_output.squeeze().numpy())
    test_data_original = scaler.inverse_transform(test_data)

    ax2.clear()
    ax2.plot(test_data_original[:, 0], label='Actual')
    ax2.plot(test_output_transformed[:, 0], label='Predicted')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Pickups')
    ax2.legend()
    ax2.set_title('Taxi Pickup Predictions vs Actual (Node 0)')

    clear_output(wait=True)
    display(plt.gcf())
    
plt.close()

# Final plot to show at the end
plt.figure(figsize=(12, 6))
plt.plot(test_data_original[:, 0], label='Actual')
plt.plot(test_output_transformed[:, 0], label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Pickups')
plt.title('Taxi Pickup Predictions vs Actual (Node 0)')
plt.legend()
plt.show()
