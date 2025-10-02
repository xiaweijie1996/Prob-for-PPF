import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import src.models.basicnetwork.transformer as transformer


# Set all tensor Double globally dtyepe
torch.set_default_dtype(torch.float64)

class CSplineBasicAttention(torch.nn.Module):
    def __init__(self, 
                 # input features
                 input_dim: int = 2,
                 
        
                 # model features transformer
                 num_blocks: int = 4,
                 emb_dim: int = 64,
                 num_heads: int = 4,
                 bias: bool = True,
                 num_nodes: int = 33,
                 num_output_nodes: int = 1,  
                 
                 # model features spline
                 b_interval: float = 5.0, # better to max of the output data maybe
                 k_bins: int = 10, # number of bins
                 
                
                 ):
        
        super(CSplineBasicAttention, self).__init__()
        
        self.input_dim = input_dim
        self.b_interval = b_interval # [-b_interval, b_interval]
        self.k_bins = k_bins
        self.num_blocks = num_blocks
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.bias = bias
        self.num_nodes = num_nodes
        self.num_output_nodes = num_output_nodes
        self.split_dim1 = 1
        
        
        # FNN takes part of the input and output K*3 -1 parameters
        self.f1 = transformer.TransformerEncoder(
            input_dim=self.input_dim,
            num_blocks=self.num_blocks,
            output_dim=self.k_bins * 3 -1,
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            bias=self.bias,
            num_nodes=self.num_nodes + input_dim//2 +1,
            num_output_nodes=self.num_output_nodes
        )
        
        
        self.f2 = transformer.TransformerEncoder(
            input_dim=self.input_dim,
            num_blocks=self.num_blocks,
            output_dim=self.k_bins * 3 -1,
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            bias=self.bias,
            num_nodes=self.num_nodes + input_dim//2 +1,
            num_output_nodes=self.num_output_nodes
        )
        # Define a special token for null condition
        self.null_token = torch.nn.Parameter(torch.randn(1, self.emb_dim))
    
    def adjusted_scaler(self, x):
        _output =  torch.tanh(x) * 2
        # if _output closer to 0, make it 1e-6
        _output = torch.where(torch.abs(_output) < 1e-10, torch.tensor(1e-10, device=x.device), _output)
        return _output
    
    
    def create_spline_params(self, params):
        """
        Convert raw network outputs into valid spline parameters.
        Args:
            params (torch.Tensor): Raw output from the network of shape (batch_size, k_bins * 3 - 1).
        Returns:
            widths (torch.Tensor): Widths of the spline bins of shape (batch_size, k_bins).
            heights (torch.Tensor): Heights of the spline bins of shape (batch_size, k_bins).
            derivatives (torch.Tensor): Derivatives at the knots of shape (batch_size, k_bins - 1).
        """
        assert params.shape[1] == self.k_bins * 3 - 1, "Invalid shape for spline parameters"
        
        batch_size = params.shape[0]
        k = self.k_bins
        B = params.shape[0]
        widths = params[:, :k]
        heights = params[:, k:2*k]
        derivatives = params[:, 2*k:]
        
        # Apply softmax to widths and heights to ensure they sum to the interval length
        widths = torch.softmax(widths, dim=-1) * (2 * self.b_interval)
        heights = torch.softmax(heights, dim=-1) * (2 * self.b_interval)

        # Apply softplus to derivatives to ensure they are positive
        derivatives = torch.nn.functional.softplus(derivatives) + 1e-3
        
        # Add zero and one to the derivatives for boundary conditions
        derivatives = torch.cat([torch.ones(batch_size, 1).to(params.device), derivatives, torch.ones(batch_size, 1).to(params.device)], dim=-1)
        
        # Cumulative sum to get knot positions
        left = -self.b_interval * torch.ones(B, 1, device=params.device, dtype=params.dtype)
        widths = torch.cat([left, left + torch.cumsum(widths, dim=-1)],  dim=-1)
        heights = torch.cat([left, left + torch.cumsum(heights, dim=-1)], dim=-1)   # (B, K+1)
        return widths, heights, derivatives
    
    def spline_transform_forward(self, input_x, widths, heights, derivatives):
        """
        Apply the spline transformation.
        Args:

            input (torch.Tensor): Input tensor of shape (batch_size, dim).
            widths (torch.Tensor): Widths of the spline bins of shape (batch_size, k_bins +1).
            heights (torch.Tensor): Heights of the spline bins of shape (batch_size, k_bins + 1 ).
            derivatives (torch.Tensor): Derivatives at the knots of shape (batch_size, k_bins + 1).
        Returns:
            output_y (torch.Tensor): Transformed output tensor of shape (batch_size, dim).
            logabsdet (torch.Tensor): Log absolute determinant of the Jacobian of shape (batch_size, dim).
        """
        input_x = input_x.clamp(widths.min()+1e-6, widths.max()-1e-6)

        # check x is in which bin from widths
        index = torch.searchsorted(widths.contiguous(), input_x.contiguous(), right=True)
        
        widths_left_value = torch.gather(widths, 1, index-1) # right value shae: (batch_size, dim)
        widths_right_value = torch.gather(widths, 1, index) # right value shae: (batch_size, dim)
        heights_left_value = torch.gather(heights, 1, index-1)
        heights_right_value = torch.gather(heights, 1, index)
        derivatives_left_value = torch.gather(derivatives, 1, index-1)
        derivatives_right_value = torch.gather(derivatives, 1, index)
        
        # Calculate the slope of the bin
        tau_x = (input_x - widths_left_value) / (widths_right_value - widths_left_value)
        s_k = (heights_right_value - heights_left_value)/(widths_right_value - widths_left_value)
        
        # Rational quadratic spline formula
        _v1 = (s_k * tau_x**2 + derivatives_left_value * tau_x * (1 - tau_x))
        _v2 = s_k + (derivatives_right_value + derivatives_left_value - 2 * s_k) * tau_x * (1 - tau_x)
        output_y = heights_left_value +  _v1 / _v2 * (heights_right_value - heights_left_value)

        # Absolute determinant of the Jacobian
        _d1 = (s_k**2 * (derivatives_right_value * tau_x**2 + 2 * s_k * tau_x * (1 - tau_x) + derivatives_left_value * (1 - tau_x)**2))
        _d2 = _v2**2
        partial_deraivatives = _d1 / _d2
        
        # Jacobian determinant
        jacobian = torch.abs(torch.cumprod(partial_deraivatives, dim=1)[:,-1])
        jacobian = jacobian.reshape(-1, 1)
       
        return output_y, jacobian, partial_deraivatives
    
    def spline_transform_reverse(self, input_y, widths, heights, derivatives):
        """
        Apply the spline transformation.
        Args:

            input_x (torch.Tensor): Input tensor of shape (batch_size, dim).
            widths (torch.Tensor): Widths of the spline bins of shape (batch_size, k_bins +1).
            heights (torch.Tensor): Heights of the spline bins of shape (batch_size, k_bins + 1 ).
            derivatives (torch.Tensor): Derivatives at the knots of shape (batch_size, k_bins + 1).
        Returns:
            output_y (torch.Tensor): Transformed output tensor of shape (batch_size, dim).
            logabsdet (torch.Tensor): Log absolute determinant of the Jacobian of shape (batch_size, dim).
        """
        input_y = input_y.clamp(heights.min()+1e-6, heights.max()-1e-6)
        
        # check x is in which bin from widths
        index = torch.searchsorted(heights.contiguous(), input_y.contiguous(), right=True)
        
        # index = torch.searchsorted(input_y.contiguous(), heights.contiguous(), right=True)
        widths_left_value = torch.gather(widths, 1, index-1) # right value shae: (batch_size, dim)
        widths_right_value = torch.gather(widths, 1, index) # right value shae: (batch_size, dim)
        heights_left_value = torch.gather(heights, 1, index-1)
        heights_right_value = torch.gather(heights, 1, index)
        derivatives_left_value = torch.gather(derivatives, 1, index-1)
        derivatives_right_value = torch.gather(derivatives, 1, index)
        
        # Calculate the slope of the bin
        s_k = (heights_right_value - heights_left_value)/(widths_right_value - widths_left_value)
        delt_y = input_y - heights_left_value
        delt_y_bin = heights_right_value - heights_left_value
        midle_value = derivatives_left_value + derivatives_right_value - 2 * s_k
        
        # Calcaute a b c
        a = delt_y * midle_value + delt_y_bin * (s_k - derivatives_left_value)
        b = delt_y_bin * derivatives_left_value - delt_y * midle_value
        c = -s_k * delt_y
        
        # Quadratic formula
        tau_x = 2 * c / (-b - torch.sqrt(b**2 - 4 * a * c))
        output_x = widths_left_value + tau_x * (widths_right_value - widths_left_value)
        
        return output_x, None, None
        
    def forward_direction(self, x, c, index_p, index_v):
        """
        Forward pass of the CSplineBasic model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            c (torch.Tensor): Condition tensor of shape (batch_size, condition_dim).
            index_p (int): Index for the positional encoding.
            index_v (float): Value for the positional encoding.
        Returns:
        """
        # Split the input tensor
        x11, x12 = x[:, :, :self.split_dim1], x[:, :, self.split_dim1:]
        
        # x2
        x21 = x11
        x11_repeat = x11.repeat(1,1,2) 
        params1 = self.f1(torch.cat([c, x11_repeat], dim=1), index_p, index_v) # shape (B, 1, k_bins*3 -1)
        params1 = params1.squeeze(1)  # shape (B, k_bins*3 -1)
        widths1, heights1, derivatives1 = self.create_spline_params(params1)
        x22, ja1, _ = self.spline_transform_forward(x12.squeeze(1), widths1, heights1, derivatives1)
        x22 = x22.unsqueeze(1)
        
        # x3
        x32 = x22
        x22_repeat = x22.repeat(1,1,2)
        params2 = self.f2(torch.cat([c, x22_repeat], dim=1), index_p, index_v) # shape (B, 1, k_bins*3 -1)
        params2 = params2.squeeze(1)  # shape (B, k_bins*
        widths2, heights2, derivatives2 = self.create_spline_params(params2)
        x31, ja2, _ = self.spline_transform_forward(x21.squeeze(1), widths2, heights2, derivatives2)
        
        # Combine the outputs
        x31 = x31.unsqueeze(1)
        x3 = torch.cat([x31, x32], dim=-1)
        ja = ja1 * ja2
        
        return x3, ja
    
    def inverse_direction(self, y, c, index_p, index_v):
        """
        Inverse pass of the CSplineBasic model.
        Args:
            y (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            c (torch.Tensor): Condition tensor of shape (batch_size, condition_dim).
            index_p (int): Index for the positional encoding.
            index_v (float): Value for the positional encoding.
        Returns:
        """
        
        # Split the input tensor
        y31, y32 = y[:, :, :self.split_dim1], y[:, :, self.split_dim1:]
        
        # x2
        y22 = y32
        y22_repeat = y22.repeat(1,1,2)
        params2 = self.f2(torch.cat([c, y22_repeat], dim=1), index_p, index_v)
        params2 = params2.squeeze(1)
        widths2, heights2, derivatives2 = self.create_spline_params(params2)
        y21, _, _ = self.spline_transform_reverse(y31.squeeze(1), widths2, heights2, derivatives2)
        y21 = y21.unsqueeze(1)
        
        # x1
        y11 = y21
        y11_repeat = y11.repeat(1,1,2)
        params1 = self.f1(torch.cat([c, y11_repeat], dim=1), index_p, index_v)
        params1 = params1.squeeze(1)
        widths1, heights1, derivatives1 = self.create_spline_params(params1)
        y12, _, _ = self.spline_transform_reverse(y22.squeeze(1), widths1, heights1, derivatives1)
        
        # Combine the outputs
        y12 = y12.unsqueeze(1)
        y1 = torch.cat([y11, y12], dim=-1)
        
        return y1, None

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = torch.randn(100, 1, 2) 
    c = torch.randn(100, 33, 2)
    y_target = torch.randn(100, 1, 2)
    index_p = 1
    index_v = 2
    
    model = CSplineBasicAttention(input_dim=2,
                         num_blocks=2, 
                         emb_dim=16, 
                         num_heads=2, 
                         bias=True, 
                         num_nodes=33, 
                         num_output_nodes=1, 
                         b_interval=5.0, 
                         k_bins=10)
    y, ja = model.forward_direction(x, c, index_p, index_v)
    print(y.shape, ja.shape)
    
    x_recon, _ = model.inverse_direction(y, c, index_p, index_v)
    print(x_recon.shape)
    
    print("Reconstruction error:", torch.mean((x - x_recon)**2).item())
    print("Allclose?", torch.allclose(x, x_recon, atol=1e-8, rtol=1e-8))
    
    # Create a small training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(500):
        optimizer.zero_grad()
        y, ja = model.forward_direction(x, c, index_p, index_v)
        loss = (y-y_target).pow(2).mean() 
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        # Plot the y and y_target
        if epoch % 20 == 0:
            plt.scatter(y.detach().numpy()[:,0,0], y.detach().numpy()[:,0,1], label='y', alpha=0.5)
            plt.scatter(y_target.detach().numpy()[:,0,0], y_target.detach().numpy()[:,0,1], label='y_target', alpha=0.5)
            plt.legend()
            plt.title(f"Epoch {epoch}")
            plt.savefig(f"src/models/mixedflow/spline_test.png")
            plt.close()
        