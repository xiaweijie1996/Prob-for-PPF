import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import src.models.basicnetwork.basicnets as basicnets
 
# Set all tensor Double globally dtyepe
torch.set_default_dtype(torch.float64)

class CSplineBasic(torch.nn.Module):
    def __init__(self, 
                 # input features
                 input_dim: int = 2,
                 hidden_dim: int = 64,
                 condition_dim: int = 128,
                 
                 # model features main
                 n_layers: int = 3,
                 split_ratio: float = 0.6,
                 b_interval: float = 5.0, # better to max of the output data maybe
                 k_bins: int = 10, # number of bins
                 
                 # model features condition
                 hidden_dim_condition: int = 32,
                 n_layers_condition: int = 3,
        
                 ):
        
        super(CSplineBasic, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.hidden_dim_condition = hidden_dim_condition
        self.n_layers_condition = n_layers_condition
        self.split_dim1 = int(input_dim * split_ratio)
        self.split_dim2 = input_dim - self.split_dim1
        self.b_interval = b_interval # [-b_interval, b_interval]
        self.k_bins = k_bins
        
        # FNN takes part of the input and output K*3 -1 parameters
        self.f1 = basicnets.BasicFFN(
            input_dim=self.split_dim1 + self.condition_dim +1, # positional encoding + null
            hidden_dim=self.hidden_dim,
            output_dim=self.k_bins * 3 -1,
            n_layers=self.n_layers
        )
        
        self.f2 = basicnets.BasicFFN(
            input_dim=self.split_dim2 + self.condition_dim +1,
            hidden_dim=self.hidden_dim,
            output_dim=self.k_bins * 3 -1,
            n_layers=self.n_layers
        )
        
        self.fc_add_dim = basicnets.BasicFFN(
            input_dim = 1,
            hidden_dim= self.hidden_dim_condition,
            output_dim = self.hidden_dim_condition,
            n_layers= self.n_layers_condition
        )
        
        self.fc_min_dim = basicnets.BasicFFN(
            input_dim = self.hidden_dim_condition,
            hidden_dim= self.hidden_dim_condition,
            output_dim = 1,
            n_layers= self.n_layers_condition
        )
        
        # Define a special token for null condition
        self.null_token = torch.nn.Parameter(torch.randn(1, self.hidden_dim_condition))

    def add_pe_and_null_to_c(self, c, index_p, index_v, postional_encoding=False):
        """
        Add positional encoding and null token to the condition vector.
        
        inpit:
        c (torch.Tensor): Condition vector of shape (batch_size, condition_dim).
        
        after self.fc_add_dim:
        c (torch.Tensor): Condition vector of shape (batch_size, hidden_dim_condition).
        
        after adding null token:
        c (torch.Tensor): Condition vector of shape (batch_size, hidden_dim_condition).
        
        after adding positional encoding:
        c_pe (torch.Tensor): Positional encoding of shape (batch_size, hidden_dim_condition).
        
        after self.fc_min_dim:
        c_add (torch.Tensor): Condition vector of shape (batch_size, 1).
        """
        #  torh.sin(index_v) and expand to match batch size
        v_info = torch.sin(torch.tensor(index_v, dtype=torch.float32)).to(c.device)
        v_info = v_info.unsqueeze(0).expand(c.shape[0], -1)  # shape (batch_size, 1)
        
        c = torch.cat([c, v_info], dim=-1) 
        c = c.unsqueeze(-1)  # shape (batch_size, condition_dim+1, 1)
        
        # Map c to hidden_dim_condition
        c_add = self.fc_add_dim(c)
        
        # Replace the index_i-th element with the null token
        c_add[:, index_p, :] = self.null_token.to(c.device)
        num_nodes = int(self.condition_dim / 2) + 1
        c_add[:, index_p + (num_nodes-1), :] = self.null_token
        
        # Add positional encoding (if transformer then use this)
        if postional_encoding:
            c_pe = basicnets.abs_pe(c_add)
            c_add = c_pe.to(c.device) + c_add
            
        # Map c_add to a single dimension
        c_add = self.fc_min_dim(c_add)
        
        return c_add.squeeze(-1)  # shape (batch_size, condition_dim +1)
    
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
        widths = torch.cumsum(widths, dim=-1) - self.b_interval
        heights = torch.cumsum(heights, dim=-1) - self.b_interval
        widths = torch.cat([-self.b_interval * torch.ones(batch_size, 1).to(params.device), widths], dim=-1)
        heights = torch.cat([-self.b_interval * torch.ones(batch_size, 1).to(params.device), heights], dim=-1)
        
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
        # check x is in which bin from widths
        index = torch.searchsorted(widths.contiguous(), input_x.contiguous(), right=True)
        # print("index:", index.shape)
        # print("max and min index:", torch.max(index), torch.min(index))
        # print("max and min input_x:", torch.max(input_x), torch.min(input_x))
        # print("max and min widths:", torch.max(widths), torch.min(widths))
        widths_left_value = torch.gather(widths, 1, index-1) # right value shae: (batch_size, dim)
        widths_right_value = torch.gather(widths, 1, index) # right value shae: (batch_size, dim)
        heights_left_value = torch.gather(heights, 1, index-1)
        heights_right_value = torch.gather(heights, 1, index)
        derivatives_left_value = torch.gather(derivatives, 1, index-1)
        derivatives_right_value = torch.gather(derivatives, 1, index)
        
        # Calculate the slope of the bin
        tau_x = (input_x - widths_left_value) / (widths_right_value - widths_left_value)
        # print("tau_x:", tau_x)
        # print("widths_left_value:", widths_left_value)
        # print("widths_right_value:", widths_right_value)
        # print("input_x:", input_x)
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
        # check x is in which bin from widths
        index = torch.searchsorted(heights.contiguous(), input_y.contiguous(), right=True)
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
        c_processed = self.add_pe_and_null_to_c(c, index_p=index_p, index_v=index_v)
        
        # Split the input tensor
        x11, x12 = x[:, :self.split_dim1], x[:, self.split_dim1:]
        
        # x2
        x21 = x11
        params1 = self.f1(torch.cat([x11, c_processed], dim=1))
        widths1, heights1, derivatives1 = self.create_spline_params(params1)
        x22, ja1, _ = self.spline_transform_forward(x12, widths1, heights1, derivatives1)
        
        # x3
        x32 = x22
        params2 = self.f2(torch.cat([x22, c_processed], dim=1))
        widths2, heights2, derivatives2 = self.create_spline_params(params2)
        x31, ja2, _ = self.spline_transform_forward(x21, widths2, heights2, derivatives2)
        
        # Combine the outputs
        x3 = torch.cat([x31, x32], dim=1)
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
        c_processed = self.add_pe_and_null_to_c(c, index_p=index_p, index_v=index_v)
        # Split the input tensor
        y31, y32 = y[:, :self.split_dim1], y[:, self.split_dim1:]
        
        # x2
        y22 = y32
        params2 = self.f2(torch.cat([y32, c_processed], dim=1))
        widths2, heights2, derivatives2 = self.create_spline_params(params2)
        y21, _, _ = self.spline_transform_reverse(y31, widths2, heights2, derivatives2)
        
        # x1
        y11 = y21
        params1 = self.f1(torch.cat([y21, c_processed], dim=1))
        widths1, heights1, derivatives1 = self.create_spline_params(params1)
        y12, _, _ = self.spline_transform_reverse(y22, widths1, heights1, derivatives1)
        
        # Combine the outputs
        y1 = torch.cat([y11, y12], dim=1)
        
        return y1, None
    
class CSplineModel(torch.nn.Module):
    def __init__(self, 
                 # input features
                 input_dim: int = 2,
                 hidden_dim: int = 64,
                 condition_dim: int = 12,
                 
                 # model features main
                 n_layers: int = 1,
                 split_ratio: float = 0.5,
                 n_blocks: int = 2,
                 
                 # model features condition
                 hidden_dim_condition: int = 32,
                 n_layers_condition: int = 2,
                 b_interval: float = 5.0,
                 k_bins: int = 10
                ):
        super(CSplineModel, self).__init__()
        
        self.blocks = torch.nn.ModuleList([
            CSplineBasic(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                n_layers=n_layers,
                split_ratio=split_ratio,
                hidden_dim_condition=hidden_dim_condition,
                n_layers_condition=n_layers_condition,
                b_interval=b_interval,
                k_bins=k_bins
            ) for _ in range(n_blocks)
        ])
    
    def forward(self, x, c, index_p, index_v):
        ja = torch.ones((x.shape[0]), device=x.device)
        for block in self.blocks:
            x, ja_block = block.forward_direction(x, c, index_p=index_p, index_v=index_v)
            ja = ja * ja_block.squeeze(-1)
        return x, ja
    
    def inverse(self, y, c, index_p, index_v):
        for block in reversed(self.blocks):
            y, _ = block.inverse_direction(y, c, index_p=index_p, index_v=index_v)
        return y, _
                         
if __name__ == "__main__":
    test_dim = 2
    c_dim = 100
    index_v = 1
    index_p = 2
    k_bins = 10
    batch_size = 400
    model = CSplineModel(
        input_dim=test_dim,
        condition_dim=c_dim,
        n_blocks=1,
        k_bins=k_bins
    )
    x = torch.randn(batch_size, test_dim)
    c = torch.randn(batch_size, c_dim)
    index_p = 3
    index_v = 0.5
    y, ja = model.forward(x, c, index_p=index_p, index_v=index_v)
    print("y:", y.shape)
    print("ja:", ja.shape)
    x_recon, _ = model.inverse(y, c, index_p=index_p, index_v=index_v)
    print("x_recon:", x_recon.shape)
    print("Difference:", (x - x_recon).mean().item())
    print( torch.allclose(x, x_recon, atol=1e-3))
    