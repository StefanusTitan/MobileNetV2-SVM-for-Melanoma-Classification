from abc import ABC, abstractmethod
import numpy as np
import numba
from scipy import sparse
from cvxopt import matrix, solvers

class BaseSVM(ABC):
    def __init__(self, C=1.0, tolerance=1e-5):
        self.C = C
        self.tolerance = tolerance
        self.a = None
        self.sv = None
        self.sv_y = None
        self.b = None

    @abstractmethod
    def kernel(self, X1, X2):
        """Metode abstrak untuk komputasi kernel."""
        pass

    def fit(self, X, y):
        n_samples = X.shape[0]
        K = self.kernel(X, X)

        # Prepare QP problem
        P = matrix(np.outer(y, y) * K)
        q = matrix(-1.0 * np.ones(n_samples))
        A = matrix(y.astype(float), (1, n_samples))
        b = matrix(0.0)

        G = spmatrix(
            [-1.0] * n_samples + [1.0] * n_samples,
            list(range(2 * n_samples)),
            list(range(n_samples)) * 2
        )
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        # Solve QP
        solvers.options['show_progress'] = True
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract support vectors
        a = np.ravel(solution['x'])
        sv_mask = a > self.tolerance
        self.a = a[sv_mask]
        self.sv = X[sv_mask]
        self.sv_y = y[sv_mask]

        # Calculate intercept
        K_sv = self.kernel(self.sv, self.sv)
        self.b = np.mean(
            self.sv_y - np.sum(self.a * self.sv_y * K_sv, axis=1)
        )

    def decision_function(self, X):
        """Hitung decision function."""
        K = self.kernel(X, self.sv)
        return np.dot(K, self.a * self.sv_y) + self.b

    def predict(self, X):
        """Perform prediction with robust input handling."""
        # Ensure X is a 2D array
        X = np.atleast_2d(X)
        
        # Compute decision values
        decision_values = self.decision_function(X)
        
        # Return predictions
        return np.sign(decision_values)

class SVM(BaseSVM):
    def kernel(self, X1, X2):
        """Kernel linier."""
        return np.dot(X1, X2.T)
    
    def calculate_weights(self): 
        """Hitung vektor bobot.""" 
        self.w = np.sum(self.a[:, None] * self.sv_y[:, None] * self.sv, axis=0) 
        
    def fit(self, X, y): 
        super().fit(X, y) 
        self.calculate_weights()

class SVM_RBF(BaseSVM):
    def __init__(self, C=1.0, gamma=0.1, chunk_size=1000, tolerance=1e-5):
        super().__init__(C, tolerance)
        self.gamma = gamma
        self.chunk_size = chunk_size

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _rbf_kernel_numba(X1, X2, gamma):
        # Ensure input arrays are properly shaped
        X1 = X1.reshape(-1, X1.shape[-1]) if X1.ndim == 1 else X1
        X2 = X2.reshape(-1, X2.shape[-1]) if X2.ndim == 1 else X2
        
        n_samples1, n_features = X1.shape
        n_samples2 = X2.shape[0]
        K = np.zeros((n_samples1, n_samples2), dtype=np.float64)

        for i in numba.prange(n_samples1):
            for j in range(n_samples2):
                dist = 0.0
                for k in range(n_features):
                    diff = X1[i, k] - X2[j, k]
                    dist += diff * diff
                K[i, j] = np.exp(-gamma * dist)
        return K

    def kernel(self, X1, X2):
        """RBF kernel dengan chunking."""
        return self._rbf_kernel_numba(X1, X2, self.gamma)