import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import pearsonr

class CCM:
    def __init__(self, X, Y, tau=1, E=2, L=500):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.tau = tau
        self.E = E
        self.L = L
        self.My = self.shadow_manifold(self.Y)
        self.t_steps, self.dists = self.get_distances(self.My)

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


    def shadow_manifold(self, X):
        X = X[:self.L]
        M = {t: [] for t in range((self.E - 1) * self.tau, self.L)}
        for t in range((self.E - 1) * self.tau, self.L):
            x_lag = [X[t - t2 * self.tau] for t2 in range(self.E)]
            M[t] = x_lag
        return M

    def get_distances(self, Mx):
        t_vec = [(k, v) for k, v in Mx.items()]
        t_steps = np.array([i[0] for i in t_vec])
        vecs = np.array([i[1] for i in t_vec])
        dists = distance.cdist(vecs, vecs, metric='euclidean')
        return t_steps, dists

    def get_nearest_distances(self, t, t_steps, dists):
        t_ind = np.where(t_steps == t)
        dist_t = dists[t_ind].squeeze()
        nearest_inds = np.argsort(dist_t)[1:self.E + 2]
        nearest_timesteps = t_steps[nearest_inds]
        nearest_distances = dist_t[nearest_inds]
        return nearest_timesteps, nearest_distances

    def predict(self, t):
        t_ind = np.where(self.t_steps == t)
        if t_ind[0].size == 0:
            # Handle case where t is not found in t_steps
            return np.nan, np.nan

        dist_t = self.dists[t_ind].squeeze()
        if dist_t.size == 0:
            # Handle case where distances array is empty
            return np.nan, np.nan

        nearest_timesteps, nearest_distances = self.get_nearest_distances(t, self.t_steps, self.dists)
        if nearest_distances.size == 0:
            # Handle case where nearest distances array is empty
            return np.nan, np.nan

        u = np.exp(-nearest_distances / np.max([1e-6, nearest_distances[0]]))
        w = u / np.sum(u)
        X_true = self.X[t]
        X_cor = np.array(self.X)[nearest_timesteps]
        X_hat = (w * X_cor).sum()
        return X_true, X_hat


    def causality(self):
        X_true_list = []
        X_hat_list = []
        for t in list(self.My.keys()):
            X_true, X_hat = self.predict(t)
            X_true_list.append(X_true)
            X_hat_list.append(X_hat)
        X_true_array = np.array(X_true_list)
        X_hat_array = np.array(X_hat_list)
        mse = np.mean((X_true_array - X_hat_array) ** 2)
        return mse

    def visualize_cross_mapping(self):
        plt.figure(figsize=(12, 6))
        for t in np.random.choice(list(self.My.keys()), size=3, replace=False):
            Ma_t = self.My[t]
            near_t, _ = self.get_nearest_distances(t, self.t_steps, self.dists)
            plt.scatter(Ma_t[0], Ma_t[1], c='b', marker='s')
            for i in range(self.E + 1):
                B_t = self.My[near_t[i]][0]
                B_lag = self.My[near_t[i]][1]
                plt.scatter(B_t, B_lag, c='r', marker='*', s=50)
                plt.plot([Ma_t[0], B_t], [Ma_t[1], B_lag], c='r', linestyle=':')
        plt.title('Cross Mapping')
        plt.xlabel('X(t)')
        plt.ylabel('X(t-1)')
        plt.show()
