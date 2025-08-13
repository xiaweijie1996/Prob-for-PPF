import torch
import numpy as np

from sklearn.mixture import GaussianMixture

# set all np to be float64
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

def inverse_pdf_gaussian(
    y: torch.Tensor,
    x_distribution: GaussianMixture,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu")
):
    """
    inverse_pdf_gaussian — what it does

    Given an invertible mapping f: R^D → R^D and a base random variable 
    X ~ p_X (here modeled by a single-component GaussianMixture), the function 
    computes the push-forward density

        p_Y(y) = p_X(f^{-1}(y)) * | det(∂f^{-1} / ∂y) |

    at given points y. It returns both log p_Y(y) and p_Y(y), along with the 
    inverse points x = f^{-1}(y).

    Inputs
    ------
    y : torch.Tensor, shape (B, D)
        Query points in Y-space.
    x_distribution : sklearn.mixture.GaussianMixture
        Must have n_components = 1; models p_X.
    model : object with .inverse(y) -> (x, det_term)
        - x is f^{-1}(y) with shape (B, D)
        - det_term encodes |det(∂f^{-1} / ∂y)| (absolute Jacobian determinant)
    device : torch.device
        Device for computation.

    Outputs
    -------
    log_p_y : np.ndarray, shape (B,)
        log p_Y(y)
    p_y : np.ndarray, shape (B,)
        p_Y(y)
    x : torch.Tensor, shape (B, D)
        Inverse points f^{-1}(y)

    """
    gmm = x_distribution
    
    # Check if gmm is one component gaussian
    if gmm.n_components != 1:
        raise ValueError("The Gaussian Mixture Model must have exactly one component for this function to work correctly.")
    
    model.eval()
    with torch.no_grad():
        y = y.to(device)
        x, det_term = model.inverse(y)  # det_term: det|∂f^{-1}/∂y| or its log

        logabsdet_inv = torch.log(det_term.abs() + 1e-12)
        logabsdet_inv = logabsdet_inv.cpu().numpy()  # Convert to numpy for GMM compatibility

        # Gaussian log p_X(x): use torch.distributions for stability
        log_p_x  = gmm.score_samples(x.cpu().numpy())  # shape (B,)
        
        log_p_x = log_p_x.reshape(-1)  # ensure it's a 1D tensor
        logabsdet_inv = logabsdet_inv.reshape(-1)  # ensure it's a 1D tensor

        log_p_y = log_p_x + logabsdet_inv  # change-of-variables
        p_y = np.exp(log_p_y)  # p_Y(y)
        
    return log_p_y, p_y, x

class CubeModel(torch.nn.Module):
    def inverse(self, y):
        x = torch.sign(y) * torch.abs(y) ** (1/3)
        det_term = 1.0 / (3.0 * torch.abs(y) ** (2/3))
        return x, det_term

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    model = CubeModel()
    device = torch.device("cpu")
    
    # Generate some test data
    np.random.seed(42)
    x = np.random.normal(0, 1, 500)  # Standard normal
    
    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(x.reshape(-1, 1))  # Reshape for GMM
    
    # y grid for evaluation
    y_vals = torch.linspace(-5, 5, 100)
    
    # Numerical PDF from inverse_pdf_gaussian
    log_p_y, p_y, _m = inverse_pdf_gaussian(
        y=y_vals.unsqueeze(1),
        model=model,
        x_distribution=gmm,
        device=device
    )

    # Analytical PDF for comparison
    def phi(z):  # standard normal pdf
        return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * z**2)

    y_np = y_vals.cpu().numpy()
    print(y_np.shape, p_y.shape)
    p_y_analytical = (1/3.0) * np.abs(y_np) ** (-2/3) * phi(np.sign(y_np) * np.abs(y_np) ** (1/3))

    # Plot PDF
    plt.figure()
    plt.plot(y_np, p_y, color='blue', alpha=0.2)
    plt.plot(y_np, p_y_analytical, '--', label="Analytical", color='red')
    plt.ylim(0, 1)
    plt.legend()
    plt.title("PDF comparison for y = x^3, X ~ N(0,1)")
    plt.savefig("figures/pdf_comparison.png")

    # Plot CDF (numerical integration)
    cdf_num = np.cumsum(p_y) * (y_np[1] - y_np[0])
    cdf_ana = np.cumsum(p_y_analytical) * (y_np[1] - y_np[0])
    plt.figure()
    plt.plot(y_np, cdf_num / cdf_num[-1], label="Numerical CDF from inverse_pdf_gaussian")
    plt.plot(y_np, cdf_ana / cdf_ana[-1], '--', label="Analytical CDF")
    plt.legend()
    plt.title("CDF comparison for y = x^3")
    plt.savefig("figures/cdf_comparison.png")