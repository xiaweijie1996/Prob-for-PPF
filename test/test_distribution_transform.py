import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from src.models.cnice.cnicemodel import CNicemModel

if __name__ == "__main__":
    # Test CNiceModelBasic
    test_dim = 2
    c_dim = 10
    index_v = 1
    index_p = 1
    batch = 1000
    
    nicem_model = CNicemModel(input_dim=test_dim, n_layers=1, split_ratio=0.5, n_blocks=2, 
                            hidden_dim=64, condition_dim=c_dim, 
                            hidden_dim_condition=32, output_dim_condition=1, n_layers_condition=2)
    
    # define a multi variant gaussian distribution
    # x_dix = torch.distributions.multivariate_normal.MultivariateNormal(
    #     loc=torch.zeros(test_dim), 
    #     covariance_matrix=torch.eye(test_dim)
    # )
    
    # x = x_dix.sample((batch,))  # Sample 5 points from the distribution
    
    # define a Gaussian Mixture Model
    _x1, _x2, x3 = np.random.rand(batch, test_dim) * 2, np.random.rand(batch, test_dim) * 3+1, np.random.rand(batch, test_dim) * 1.2+5
    _x = np.concatenate((_x1, _x2, x3), axis=0)
    gmm = GaussianMixture(n_components=4, covariance_type='full')
    gmm.fit(_x)
    x = torch.tensor(gmm.sample(batch)[0], dtype=torch.float32)  # Sample from GMM
    
    # Plot distribution of x
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), alpha=0.5)
    plt.title('Sampled Points from GMM')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('test/gmm_sampled_points.png')
    
    c = torch.ones(batch, c_dim)  # Condition vector
    
    # input = torch.cat((x, x), dim=1)
    # print("Input shape:", input.shape)
    output, _ja = nicem_model.forward(x, c, index_p=index_p, index_v=index_v)
    print("Output shape:", output.shape)
    print("Jacobian determinant shape:", _ja.shape)
    
    # f(x) = y
    # p(x) = p(y) * |det(d f^-1(y)/ dy)|
    # Plot emperical pdf and cdf of y
    # Convert output to numpy
    y = output.detach().cpu().numpy()

    # Check the min and max if x and y
    max_y0, min_y0 = y[:,0].max().item(), y[:,0].min().item()
    max_y1, min_y1 = y[:,1].max().item(), y[:,1].min().item()
    
    n_bins = int(np.sqrt(batch))  # Number of bins for the histogram
    y0_line = np.linspace(min_y0, max_y0, n_bins)
    y1_line = np.linspace(min_y1, max_y1, n_bins)
    gap_area = (max_y0 - min_y0) * (max_y1 - min_y1) / (n_bins * n_bins)
    
    grid_yy0, grid_yy1 = np.meshgrid(y0_line, y1_line)
    grid_y0, grid_y1 = np.meshgrid(y0_line, y1_line, sparse=True)
    density = np.zeros((n_bins, n_bins))
    cum_density = np.zeros((n_bins, n_bins))
    print("Grid shape:", grid_y0.shape, grid_y1.shape)
    # print(grid_y0)
    # print(grid_y1)
    # compute the density of each bin  
    for i in range(n_bins):
        for j in range(n_bins):
            # pdf filter
            filter = (
            (y[:, 1] >= grid_y1[j, 0]) &
            (y[:, 1] <  grid_y1[j, 0] + (max_y1 - min_y1) / n_bins) &  # y in y-bin j
            (y[:, 0] >= grid_y0[0, i]) &
            (y[:, 0] <  grid_y0[0, i] + (max_y0 - min_y0) / n_bins)    # x in x-bin i
            )
    
            density[j, i] = np.sum(filter) / (batch*gap_area)
            
            # cdf filter
            cdf_filter = (
                (y[:, 1] <= grid_y1[j, 0]) &
                (y[:, 0] <= grid_y0[0, i])
            )
            cum_density[j, i] = np.sum(cdf_filter) / batch
            
    # Plot the pdf and cdf in a 3d plot
    fig = plt.figure(figsize=(14, 6))

    # First subplot: PDF
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(grid_yy0, grid_yy1, density, cmap='viridis', edgecolor='none')
    ax1.set_title('PDF of Output')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Density')

    # Second subplot: CDF
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(grid_yy0, grid_yy1,  cum_density, cmap='viridis', edgecolor='none')
    ax2.set_title('CDF of Output')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Cumulative Density')

    plt.tight_layout()
    plt.savefig('test/output_pdf_cdf.png')
    plt.close()
        
    # Check the inverse function
    x_inverse, _ja_inverse = nicem_model.inverse(output, c, index_p=index_p, index_v=index_v)
    # check if the inversed out is the same as the input
    print("Are the original and inversed outputs close?", torch.allclose(x, x_inverse, atol=1e-6))
    # p_y_compute = x_dix.log_prob(x_inverse).exp() * _ja_inverse

    density_y = torch.zeros((n_bins, n_bins))
    _input_y = torch.zeros((n_bins* n_bins, 2))
    _batch_index = 0
    for i in range(n_bins):
        for j in range(n_bins):
            _y0 = grid_y0[0, i]
            _y1 = grid_y1[j, 0]
            _input_y[_batch_index] = torch.tensor([_y0, _y1])
            _batch_index += 1
    print("Input y shape:", _input_y.shape)
    c = torch.ones(n_bins* n_bins, c_dim)  # Condition vector for inverse
    x_inverse, _ja_inverse = nicem_model.inverse(_input_y, c, index_p=index_p, index_v=index_v)
    
    # p_y_compute = x_dix.log_prob(x_inverse).exp() * _ja_inverse
    p_y_compute = gmm.score_samples(x_inverse.detach().numpy())
    p_y_compute = torch.tensor(p_y_compute, dtype=torch.float32)
    p_y_compute = p_y_compute.exp()* _ja_inverse
    print(_ja_inverse.mean().item(), p_y_compute.mean().item())
    _batch_index = 0
    for i in range(n_bins):
        for j in range(n_bins):
            density_y[i, j] = p_y_compute[_batch_index].item()
            _batch_index += 1
    
    # plot the density of the output
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(grid_yy0, grid_yy1, density_y, cmap='viridis', edgecolor='none')
    plt.title('Density of Output')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('test/output_density.png')
    plt.close()
