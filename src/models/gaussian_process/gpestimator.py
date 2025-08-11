"""
Gaussian-process emulator for (P,Q) -> (|V|, angle) mapping.

This wrapper uses scikit-learn's GaussianProcessRegressor (GPR) to emulate an
AC power-flow black box. Inputs X are stacked active/reactive injections and
targets y are stacked voltage magnitudes and angles.

Key points:
- X shape: (n_samples, n_features) where n_features = 2 * (n_nodes - 1)
           (all loads' P then all loads' Q)
- y shape: (n_samples, n_targets) where n_targets = 2 * (n_nodes - 1)
           (all buses' |V| then all buses' angle), or whatever your transforms return
- Multi-output: scikit-learn fits **independent** GPs per output column, sharing
  the same kernel hyperparameters across outputs (no cross-output covariance).
- Kernel: additive multi-scale RBF (short + long) + white noise, with **separated
  bounds** to discourage both RBFs collapsing to the same length-scale.
"""

import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from src.powersystems.randomsys import randomsystem, magnitude_transform, angle_transform

class GPestimator:
    """
    Gaussian-process estimator for power-flow emulation.

    This class learns f: X -> y from samples (X, y), where:
      - X = [P_1..P_L, Q_1..Q_L] (features per sample), shape (n_samples, 2L)
      - y could be concatenated [|V|_1..|V|_B, angle_1..angle_B], shape (n_samples, 2B)

    Notes
    -----
    - scikit-learn's GPR handles multi-output by fitting **one GP per target column**,
      sharing kernel hyperparameters; it does not model cross-output correlation.
    - The kernel used here is k = k_short + k_long + white. The RBF components have
      **separated length-scale bounds** to avoid degeneracy (both RBFs learning the
      same scale).
    """
    def __init__(self) -> None: 
        pass
    
    def fit(self, 
            X, 
            y, 
            random_state=21,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=20,
            normalize_y=True):
        
        """
        Fit the Gaussian Process model to training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features for each sample. For a network with L loads,
            n_features should typically be 2L (all P then all Q).
        y : np.ndarray, shape (n_samples, n_targets)
            Target values (e.g., stacked |V| and angle for all non-slack buses).
            Multi-output is supported; sklearn fits independent outputs.
        random_state : int, default=21
            Seed passed to the optimizer (for reproducible restarts).

        Returns
        -------
        self : GPestimator
            Fitted estimator (allows chaining).

        Notes
        -----
        Kernel choice:
          k_short = C * RBF(ℓ ∈ [1e-2, 0.8])   # short-scale variation
          k_long  = C * RBF(ℓ ∈ [2.0, 50.0])   # long-scale variation
          white   = White noise for numerical stability
        The separated ℓ-bounds intentionally break symmetry so the two RBFs
        don't collapse to the same scale on small datasets.
        """
        k1 = C(1.0,(1e-3,1e3)) * RBF(0.3, length_scale_bounds=(1e-2, 0.8))
        k2 = C(1.0,(1e-3,1e3)) * RBF(6.0, length_scale_bounds=(2.0, 50.0))
        kernel = k1 + k2 + WhiteKernel(1e-5,(1e-9,1e-2))

        self.gp = GaussianProcessRegressor(kernel=kernel, 
                                            n_restarts_optimizer=n_restarts_optimizer,
                                            normalize_y=normalize_y,
                                            random_state=random_state,
                                            optimizer=optimizer)
        self.gp.fit(X, y)

    def predict(self, X):
        """
        Predict mean and covariance at new inputs.

        Parameters
        ----------
        X : np.ndarray, shape (n_test, n_features)
            Test inputs. Must have the **same number of features** as training X.

        Returns
        -------
        mean : np.ndarray, shape (n_test, n_targets)
            Posterior predictive mean for each output dimension.
        cov  : np.ndarray, shape (n_test, n_test, n_targets)
            For each target dimension d, cov[:, :, d] is the n_test×n_test
            covariance matrix across test points. Cross-output covariances are
            not modeled (independent outputs), hence no (n_test, n_test, Dy, Dy).

        Raises
        ------
        RuntimeError
            If called before .fit().
        """
        if not hasattr(self, 'gp'):
            raise RuntimeError("The model has not been fitted yet.")
        return self.gp.predict(X, return_cov=True)
    
    def sample(self, X, n_samples=100):
        """
        Sample from the posterior distribution at new inputs.

        Parameters
        ----------
        X : np.ndarray, shape (n_test, n_features)
            Test inputs. Must have the **same number of features** as training X.
        n_samples : int, default=100
            Number of samples to draw from the posterior.

        Returns
        -------
        samples : np.ndarray, shape (n_samples, n_test, n_targets)
            Samples from the posterior distribution for each output dimension.

        Raises
        ------
        RuntimeError
            If called before .fit().
        """
        if not hasattr(self, 'gp'):
            raise RuntimeError("The model has not been fitted yet.")
        return self.gp.sample_y(X, n_samples=n_samples)
    
    @property
    def _get_inital_params(self):
        """
        Get the parameters of the Gaussian Process model.
        
        Returns:
        dict: A dictionary containing the parameters of the model.
        """
        if not hasattr(self, 'gp'):
            raise RuntimeError("The model has not been fitted yet.")
        return self.gp.kernel
    
    @property
    def _get_current_params(self):
        """
        Get the current parameters of the Gaussian Process model.
        
        Returns:
        dict: A dictionary containing the current parameters of the model.
        """
        if not hasattr(self, 'gp'):
            raise RuntimeError("The model has not been fitted yet.")
        return self.gp.kernel_

if __name__ == "__main__":
    
    # Import the random system
    n_nodes = 1000
    total_samples = 100
    active_power = np.random.normal(30, scale=5, size=(total_samples, n_nodes-1))  + np.random.normal(20, scale=4, size=(total_samples, n_nodes-1))  # Power in kW
    reactive_power = np.random.normal(10, scale=2, size=(total_samples, n_nodes-1)) + np.random.normal(1, scale=2, size=(total_samples, n_nodes-1))  # Reactive power in kVAR
    
    system = randomsystem(num_nodes=n_nodes, num_children=5)
    result = system.run(
        active_power=active_power,
        reactive_power=reactive_power,
        plot_graph=False
    )
    
    print("Voltage magnitudes:", result["v"].shape)
    
    # Fit the Gaussian Process model
    n_samples = int(total_samples * 0.8)
    X = np.hstack((active_power[:n_samples,:], reactive_power[:n_samples,:]))
    y_mag = magnitude_transform(result["v"][:n_samples, :])
    y_angle = angle_transform(result["v"][:n_samples, :])
    
    # Combine real and imaginary parts
    y = np.hstack((y_mag, y_angle))
    
    gp_estimator = GPestimator()
    gp_estimator.fit(X, y)
    
    # Predict using the fitted model
    X_test = np.hstack((active_power[n_samples:, :], reactive_power[n_samples:, :]))
    mean, cov = gp_estimator.predict(X_test)
    
    y_target_mag = magnitude_transform(result["v"][n_samples:, :])
    y_target_angle = angle_transform(result["v"][n_samples:, :])
    y_target = np.hstack((y_target_mag, y_target_angle))
    
    # print("errors:", (np.abs(mean - y_target)/ np.abs(y_target)).mean() * 100)
    print("errors (magnitude):", (np.abs(mean[:, :n_nodes-1] - y_target_mag) / np.abs(y_target_mag)).mean() * 100)
    print("errors (angle):", (np.abs(mean[:, n_nodes-1:] - y_target_angle) / np.abs(y_target_angle)).mean() * 100)

    # Print the parameters of the model
    print("Model parameters:", gp_estimator._get_inital_params)
    print("Current kernel parameters:", gp_estimator._get_current_params)
    
    # Print STD and COV
    # # print("Standard Deviation of predictions:", std)
    print("Covariance of predictions:", cov.shape)
    print("Mean of predictions:", mean.shape)