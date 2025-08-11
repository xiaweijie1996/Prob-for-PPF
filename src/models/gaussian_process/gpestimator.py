import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

import src.powersystems.randomsystem as randsys
from sklearn.gaussian_process.kernels import RBF


class GPestimator:
    """
    A class to estimate

    """
    def __init__(self):
        pass
    
    def fit(self, X, y):
        """
        Fit the Gaussian Process model to the data.
        
        Parameters:
        X (np.ndarray): Input features.
        y (np.ndarray): Target values.
        """
        kernel = RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel)
        self.gp.fit(X, y)

    def predict(self, X):
        """Predict using the fitted Gaussian Process model.
        """
        if not hasattr(self, 'gp'):
            raise RuntimeError("The model has not been fitted yet.")
        return self.gp.predict(X)
    
    def _get_params(self):
        """
        Get the parameters of the Gaussian Process model.
        
        Returns:
        dict: A dictionary containing the parameters of the model.
        """
        if not hasattr(self, 'gp'):
            raise RuntimeError("The model has not been fitted yet.")
        return self.gp.get_params()
    

if __name__ == "__main__":
    
    # Import the random system
    n_nodes = 3
    active_power = np.random.normal(50, scale=2, size=(10, n_nodes-1))  # Power in kW
    reactive_power = np.random.normal(10, scale=2, size=(10, n_nodes-1))  # Reactive power in kVAR
    
    system = randsys.randomsystem(num_nodes=n_nodes, num_children=1)
    result = system.run(
        active_power=active_power,
        reactive_power=reactive_power,
        plot_graph=True
    )
    
    print("Voltage magnitudes:", result["v"].shape)
    
    
    # Fit the Gaussian Process model
    n_samples = 5
    X = np.hstack((active_power[:n_samples,:], reactive_power[:n_samples,:]))
    y_real = result["v"].real[:n_samples, :]
    y_imag = result["v"].imag[:n_samples, :]
    
    # Combine real and imaginary parts
    y = np.hstack((y_real, y_imag))
    
    gp_estimator = GPestimator()
    gp_estimator.fit(X, y)
    
    # Predict using the fitted model
    X_test = np.hstack((active_power[n_samples:, :], reactive_power[n_samples:, :]))
    predictions = gp_estimator.predict(X_test)
    
    y_target_real = result["v"].real[n_samples:, :]
    y_target_imag = result["v"].imag[n_samples:, :]
    y_target = np.hstack((y_target_real, y_target_imag))
    
    print("Predictions:", predictions)
    print("True values:", y_target)