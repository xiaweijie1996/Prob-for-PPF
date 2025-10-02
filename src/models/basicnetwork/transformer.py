import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Batch-first self-attention (MHA) with fused QKV projection.

    Args:
        embed_dim (int): Channel size.
        num_heads (int): Number of heads (embed_dim must be divisible by num_heads).
        bias (bool): Use bias in linear layers.
        graph_info (torch.Tensor): using graph information for attention mask. this is a (num_nodes, num_nodes) tensor.
    """
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, graph_info: torch.Tensor = None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.graph_info = graph_info.unsqueeze(0).unsqueeze(0) # a adjacency matrix of shape (num_nodes, num_nodes) with 1 for connected nodes and 0 for unconnected nodes

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, C) -> (B, H, S, Dh)
        B, S, _ = x.size()
        x = x.view(B, S, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # (B, H, S, Dh)
        return x

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> torch.Tensor:
        B, s_len, e_d = x.size()
        assert e_d == self.embed_dim, f"Expected input with {self.embed_dim} channels, got {e_d}"

        # Fused projection, then split
        qkv = self.qkv(x)                      # (B, S, 3C)
        q, k, v = qkv.chunk(3, dim=-1)         # each (B, S, C)

        # Heads
        qh = self._shape(q)                    # (B, H, S, Dh)
        kh = self._shape(k)                    # (B, H, S, Dh)
        vh = self._shape(v)                    # (B, H, S, Dh)

        # Attention
        attn_logits = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        # Apply graph info mask if provided
        if self.graph_info is not None:
            # make zero  in graph_info to -inf
            mask = self.graph_info.to(x.device)  # (1, 1, S, S)
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
            # print(attn_logits)
            attn = F.softmax(attn_logits, dim=-1)  # (B, H, S, S)
        else:
            attn = F.softmax(attn_logits, dim=-1)  # (B, H, S, S)
        
        # Aggregate
        out_heads = torch.matmul(attn, vh)     # (B, H, S, Dh)

        # Merge heads
        out = out_heads.transpose(1, 2).contiguous().view(B, s_len, self.embed_dim)  # (B, S, C)
        out = self.out_proj(out)

        if need_weights:
            return out, attn                   # (B, S, C), (B, H, S, S)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 ff_hidden_dim: int,
                 graph_info: torch.Tensor,
                 bias: bool = True):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, bias=bias, graph_info=graph_info)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim, bias=bias),
            nn.ReLU(),
        )
        self.ff_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Multi-head attention
        attn_out = self.attn(x)
        x = x + attn_out
        x = self.attn_layer_norm(x)

        # Feed-forward
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.ff_layer_norm(x)
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, 
                # Encoder features
                input_dim: int,
                num_blocks: int,
                output_dim: int,
                
                # Attention features
                embed_dim: int,
                num_heads: int,
                
                # Default features
                num_nodes: int = 33,  # Not used in current implementation
                num_output_nodes: int = 1,  # Not used in current implementation
                bias: bool = True,
                graph_info: torch.Tensor = None,  # adjacency matrix of shape (num_nodes, num_nodes)
                ):
        super().__init__()
        # Initialize parameters
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        
        # Add Null token
        self.null_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_add_dim = nn.Linear(input_dim, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_dim=embed_dim * 4,
                graph_info=graph_info,
                bias=bias
            ) for _ in range(num_blocks)
        ])
        
        # Dense layer to output_dim
        self.fc_out_feature = nn.Linear(embed_dim, output_dim)
        self.fc_out_node = nn.Linear(num_nodes, num_output_nodes)  
        
        # Input_dim has to be 2
        assert input_dim == 2, "Input feature dimension must be 2"
        
        # Assert embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
    def add_null_to_c(self, c, index_p, index_v):
        """
        c: (batch_size, s_len, condition_dim) eg. (B, 33, 2)
        
        Add a null token to c at index_p position, and add v_info to c.
        Return c_add: (batch_size, condition_dim + 1, hidden_dim_condition)
        """
        #  Add v_info to c as a information of for model to know which v the target
        v_info = torch.tensor(index_v).to(c.device)
        v_info0 = torch.sin(v_info * math.pi / 180.0)  # shape (1,)
        v_info1 = torch.cos(v_info * math.pi / 180.0)  # shape (1,)
        v_info = torch.stack([v_info0, v_info1], dim=-1)  # shape (2,)
        v_info = v_info.unsqueeze(0).expand(c.shape[0], -1, -1) # shape (batch_size, 1, 2)
        
        c = torch.cat([c, v_info], dim=1)  # shape (batch_size, condition_dim+1, 2)

        # Map c to hidden_dim_condition
        c_add = self.fc_add_dim(c) # shape (batch_size, condition_dim +1, hidden_dim_condition)
        
        # Replace the index_i-th element with the null token for model to know which p is given
        c_add[:, index_p, :] = self.null_token.to(c.device) 
      
        return c_add # shape (batch_size, condition_dim +1, hidden_dim_condition)
    
    def add_positional_encoding(self, c):
        """
        c: (batch_size, s_len, embed_dim) eg. (B, 33, 16)
        
        Add positional encoding to c.
        Return c_pe: (batch_size, s_len, embed_dim)
        """
        batch_size, s_len, embed_dim = c.size()
        pe = torch.zeros(s_len, embed_dim, device=c.device)
        position = torch.arange(0, s_len, dtype=torch.float, device=c.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=c.device).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, s_len, embed_dim)
        c_pe = c + pe
        return c_pe

    def forward(self, c, index_p, index_v):
        # Add null token to c
        c = self.add_null_to_c(c, index_p, index_v) # shape (batch_size, condition_dim +1, hidden_dim_condition)
        
        # Add positional encoding to c
        c = self.add_positional_encoding(c) # shape (batch_size, condition_dim +1, embed_dim)
        
        # Transformer blocks
        for block in self.blocks:
            c = block(c)  # (B, s_len, embed_dim)
        
        # Output layer
        c_out = self.fc_out_feature(c)  # (B, s_len, output_dim)
        c_out = self.fc_out_node(c_out.transpose(1, 2)).transpose(1, 2)  # (B, num_nodes, output_dim) -> (B, num_output_nodes, output_dim)       
        return c_out
        


# ---------- Example usage ----------
if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(2, 33, 2)  # (B, T, C)
    B, T, C = x.size()
    H = 2
    graph_info = torch.randint(0, 2, (T+1, T+1))  # Random adjacency matrix for example
   
    # Transformer Encoder
    encoder = TransformerEncoder(
        input_dim=2,
        num_blocks=2,
        output_dim=16,
        embed_dim=16,
        num_heads=H,
        bias=True,
        num_nodes=34, # because we add one null token
        num_output_nodes=1,
        graph_info=graph_info
        
    )
    index_p = 1
    index_v = 1
    c_out = encoder(x, index_p=index_p, index_v=index_v)  # (B, T, output_dim)
    print(c_out.shape)  # should be (2, 33, 16)
