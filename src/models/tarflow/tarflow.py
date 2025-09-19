import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Batch-first multi-head attention (MHA), implemented from scratch.

    Args:
        embed_dim (int): Input/Output feature size (channels).
        num_heads (int): Number of attention heads. Must divide embed_dim.
        attn_dropout (float): Dropout on attention probabilities.
        proj_dropout (float): Dropout on the output projection.
        bias (bool): Whether to use bias in the linear projections.

    Shapes:
        q, k, v: (B, T_q, C), (B, T_k, C), (B, T_k, C)
        mask (optional): bool or float mask that broadcasts to (B, 1, T_q, T_k)
            - If bool: True for positions to KEEP, False to MASK OUT
            - If float: additively applied to the attention logits (e.g., -inf on masked)
        Returns:
            out: (B, T_q, C)
            attn (optional): (B, num_heads, T_q, T_k) if need_weights=True
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # dim per head
        self.scale = self.head_dim ** -0.5

        # One fused projection for qkv is common & fast
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, input: torch.Tensor):
        # [B, S_len, embed_dim] -> [B, num_heads, S_len, head_dim]
        B, s_len, e_d = input.size()
        input = input.view(B, s_len, self.num_heads, self.head_dim)
        input = input.transpose(1, 2)  # (B, num_heads, S_len, head_dim)
        return input

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
    ):
        B, s_len, e_d = x.size() # [B, S_len, embed_dim]
        assert C == self.embed_dim, f"Expected input with {self.embed_dim} channels, got {C}"
        
        qkv = self.qkv(x)  # (B, s_len, 3*embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # (B, s_len, embed_dim) each
        
        # Reshape and transpose for multi-head attention
        qh = self._shape(q)  # (B, H, s_len, head_dim)
        kh = self._shape(k)  # (B, H, s_len, head_dim)
        vh = self._shape(v)  # (B, H, s_len, head_dim)
        
        # Attention scores: (B, H, s_len, head_dim) @ (B, H, head_dim, s_len) * scale-> (B, H, s_len, s_len)
        attn_logits = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale

        # Softmax over key positions
        attn = F.softmax(attn_logits, dim=-1)
        
        # Weighted sum: (B, H, s_len, s_len) @ (B, H, s_len, head_dim) -> (B, H, s_len, head_dim)
        out_heads = torch.matmul(attn, vh)

        # Merge heads: (B, H, s_len, head_dim) -> (B, s_len, H, head_dim) -> (B, s_len, embed_dim)
        out = out_heads.transpose(1, 2).contiguous().view(B, Tq, C)
        out = self.out_proj(out)
        
        if need_weights:
            return out, attn  # (B, s_len, embed_dim), (B, H, s_len, s_len)
        return out


# ---------- Example usage ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    
    x = torch.randn(2, 10, 32)  # (B, T, C)
    B, Tq, C = x.size()
    H = 4  # number of heads
    mha = MultiHeadAttention(embed_dim=C, num_heads=H)

    out, attn = mha(x, need_weights=True)
    print("Output shape:", out.shape)  # (B, T, C)
    print("Attention shape:", attn.shape)  # (B, H, T, T)