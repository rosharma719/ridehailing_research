import os
import numpy as np

# Ensure directory exists
os.makedirs("../custom_data", exist_ok=True)

adj_matrix = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 1]
], dtype=np.int8)

np.save("../custom_data/adjacency.npy", adj_matrix)
