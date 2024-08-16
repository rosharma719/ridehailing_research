import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr

class CCM:
    def __init__(self, X, Y, tau=1, E=2, L=None):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.tau = tau
        self.E = E
        self.L = L if L else len(X)
        self.Mx = self._construct_shadow_manifold(self.X)
        
        # Debugging info
        print(f"Initialized CCM")
        print(f"X shape: {self.X.shape}")
        print(f"Y shape: {self.Y.shape}")
        print(f"L: {self.L}")
        print(f"Shadow Manifold Mx size: {len(self.Mx)}")

    def _construct_shadow_manifold(self, data):
        data_len = len(data)
        data = data[:self.L]
        M = {t: [] for t in range((self.E - 1) * self.tau, min(self.L, data_len))}
        print(f"Constructing shadow manifold with data size: {data.shape} and L={self.L}")
        for t in range((self.E - 1) * self.tau, min(self.L, data_len)):
            x_lag = []
            for t2 in range(self.E):
                x_lag.extend(data[t - t2 * self.tau])  # Flatten the data
            M[t] = x_lag
        return M

    def _find_nearest_neighbors(self, t):
        t_vec = np.array([v for v in self.Mx.values()])
        dist_matrix = distance.cdist(np.array([self.Mx[t]]), t_vec, metric='euclidean').flatten()
        nearest_indices = np.argsort(dist_matrix)[1:self.E + 2]
        return nearest_indices, dist_matrix[nearest_indices]

    def _predict_y(self, t):
        nearest_indices, nearest_distances = self._find_nearest_neighbors(t)
        weights = np.exp(-nearest_distances / np.max(nearest_distances))
        weights /= np.sum(weights)
        y_pred = np.sum(weights * self.Y[nearest_indices])
        return y_pred

    def calculate_correlation(self, X_test=None, Y_test=None):
        if X_test is not None and Y_test is not None:
            print(f"Calculating correlation with test data")
            print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
            test_shadow_manifold = self._construct_shadow_manifold(X_test)
            Y_true = []
            Y_pred = []
            for t in test_shadow_manifold.keys():
                Y_true.append(Y_test[t])
                Y_pred.append(self._predict_y(t))
        else:
            print(f"Calculating correlation with training data")
            Y_true = []
            Y_pred = []
            for t in self.Mx.keys():
                Y_true.append(self.Y[t])
                Y_pred.append(self._predict_y(t))
        
        correlation, _ = pearsonr(Y_true, Y_pred)
        return correlation

class NearestNeighbors:
    def __init__(self, X_train, Y_train, n_neighbors=5):
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_neighbors = n_neighbors

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            dists = distance.cdist(self.X_train, [x], metric='euclidean').flatten()
            nearest_indices = np.argsort(dists)[:self.n_neighbors]
            nearest_y = self.Y_train[nearest_indices]
            y_pred = np.mean(nearest_y)
            predictions.append(y_pred)
        return np.array(predictions)
