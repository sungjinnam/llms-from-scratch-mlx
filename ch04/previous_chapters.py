import mlx.core as mx
import mlx.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.mask = mx.tril(mx.ones((context_length, context_length)), k=0)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # splitting the matrix by addinga num_heads dimension
        keys = keys.reshape(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.reshape(b, num_tokens, self.num_heads, self.head_dim)
        values = values.reshape(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose((0, 2, 1, 3))
        queries = queries.transpose((0, 2, 1, 3))
        values = values.transpose((0, 2, 1, 3))

        attn_scores = queries @ keys.transpose((0, 1, 3, 2))
        mask_bool = self.mask.astype(mx.bool_)[:num_tokens, :num_tokens]
        attn_scores = mx.where(mask_bool, attn_scores, -mx.inf)
        attn_weights = mx.softmax(attn_scores / keys.shape[-1]**0.5, axis=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(0, 2, 1, 3)
        # combine heads; self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
    
    def __call__(self, x):
        return self.forward(x)
