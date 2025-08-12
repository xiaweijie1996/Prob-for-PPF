import torch

def inverse_pdf_gaussian(
    y: torch.Tensor,
    mean_x: torch.Tensor,
    cov_matr: torch.Tensor,
    model,
    device: torch.device = torch.device("cpu"),
    assume_det_is_log: bool = False,
):
    """
    Compute log p_Y(y) and p_Y(y) for Y = f(X), with X ~ N(mean_x, cov_matr),
    using x = f^{-1}(y) from an invertible model (e.g., NICE).

    Parameters
    ----------
    y : (B, D) tensor
        Points in Y-space where the density is evaluated.
    mean_x : (D,) tensor
        Mean of the base Gaussian in X-space.
    cov_matr : (D, D) tensor
        Covariance of the base Gaussian in X-space (SPD).
    model : object with .inverse(y) -> (x, det_term)
        Must return the inverse transform x = f^{-1}(y) and either the
        determinant of the inverse Jacobian (det_term) or its log.
    device : torch.device
        Device to run on.
    assume_det_is_log : bool
        If True, det_term from model.inverse is already log|det|. If False,
        we take log(|det_term|) with numerical stabilisation.

    Returns
    -------
    log_pdf : (B,) tensor
        log p_Y(y).
    pdf : (B,) tensor
        p_Y(y).
    x : (B, D) tensor
        The inverse points x = f^{-1}(y).
    """
    y = y.to(device).float()
    mean_x = mean_x.to(device).float()
    cov_matr = cov_matr.to(device).float()

    model.eval()
    with torch.no_grad():
        x, det_term = model.inverse(y)  # det_term: det|∂f^{-1}/∂y| or its log
        x = x.to(device).float()

        if assume_det_is_log:
            logabsdet_inv = det_term
        else:
            # Convert determinant to log-determinant safely
            logabsdet_inv = torch.log(det_term.abs() + 1e-12)

        # Gaussian log p_X(x): use torch.distributions for stability
        mvn = torch.distributions.MultivariateNormal(loc=mean_x, covariance_matrix=cov_matr)
        log_p_x = mvn.log_prob(x)  # shape (B,)
        log_p_x = log_p_x.view(-1)  # ensure it's a 1D tensor
        logabsdet_inv = logabsdet_inv.view(-1)  # ensure it's a 1D tensor
        
        log_p_y = log_p_x + logabsdet_inv  # change-of-variables
        p_y = torch.exp(log_p_y)

    return log_p_y, p_y, x

class CubeModel(torch.nn.Module):
    def inverse(self, y):
        # y -> x
        x = torch.sign(y) * torch.abs(y) ** (1/3)
        # derivative dx/dy = 1 / (3 * |y|^(2/3))
        det_term = 1.0 / (3.0 * torch.abs(y) ** (2/3))
        return x, det_term
    

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    device = torch.device("cpu")
    model = CubeModel()

    # y grid for evaluation
    y_vals = torch.linspace(-5, 5, 100)

    # Numerical PDF from inverse_pdf_gaussian
    log_p_y, p_y, _ = inverse_pdf_gaussian(
        y=y_vals.unsqueeze(1),
        mean_x=torch.zeros(1),
        cov_matr=torch.eye(1),
        model=model,
        device=device,
        assume_det_is_log=False
    )

    # Analytical PDF for comparison
    def phi(z):  # standard normal pdf
        return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * z**2)

    y_np = y_vals.numpy()
    print(y_np.shape, p_y.shape)
    p_y_analytical = (1/3.0) * np.abs(y_np) ** (-2/3) * phi(np.sign(y_np) * np.abs(y_np) ** (1/3))

    # Plot PDF
    plt.figure()
    plt.plot(y_np, p_y.numpy(), color='blue', alpha=0.2)
    plt.plot(y_np, p_y_analytical, '--', label="Analytical", color='red')
    plt.ylim(0, 1)
    # plt.legend()
    plt.title("PDF comparison for y = x^3, X ~ N(0,1)")
    plt.savefig("figures/pdf_comparison.png")

    # Plot CDF (numerical integration)
    cdf_num = np.cumsum(p_y.numpy()) * (y_np[1] - y_np[0])
    cdf_ana = np.cumsum(p_y_analytical) * (y_np[1] - y_np[0])
    plt.figure()
    plt.plot(y_np, cdf_num / cdf_num[-1], label="Numerical CDF from inverse_pdf_gaussian")
    plt.plot(y_np, cdf_ana / cdf_ana[-1], '--', label="Analytical CDF")
    plt.legend()
    plt.title("CDF comparison for y = x^3")
    plt.savefig("figures/cdf_comparison.png")