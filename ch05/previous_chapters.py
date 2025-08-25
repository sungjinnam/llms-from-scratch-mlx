import mlx.core as mx
import mlx.nn as nn
import mlx.data as dx

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def __call__(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(mx.arange(seq_len))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def __call__(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # shortcut connection for feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = mx.ones(emb_dim)
        self.shift = mx.zeros(emb_dim)
    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        norm_x = (x - mean) / mx.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    def __call__(self, x):
        return self.forward(x)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def __call__(self, x):
        return self.layers(x)
    
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
    
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens)array of indices in the current context.
    for _ in range(max_new_tokens):
        # get current context within the context window
        idx_cond = idx[:, -context_size:]
        # get the predictions
        logits = model(mx.stop_gradient(idx_cond))
        # focus on the last time step
        logits = logits[:, -1, :]
        probas= mx.softmax(logits, axis=-1)
        # get the idx of the vocab entry with the highest probability
        idx_next = mx.argmax(probas, axis=-1, keepdims=True)
        # append sampled index to the sequence
        idx = mx.concat([idx, idx_next], axis=1)
    return idx
        
class GPTDatasetV1:
    def __init__(self, txt, tokenizer, max_length, stride, batch_size, shuffle=True):
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # sliding window to chunk the input text with overlaps
        self.chunks = []
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.chunks.append({
                "input_ids": input_chunk,
                "target_ids": target_chunk
            })

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self):
        # creates an independent data stream each time it is called.
        stream = dx.buffer_from_vector(self.chunks)
        stream = stream.to_stream()
        if self.shuffle:
            stream = stream.shuffle(buffer_size=len(self.chunks))
        stream = stream.batch(self.batch_size)
        return stream

    def __len__(self):
        return len(self.chunks)

    def __iter__(self):
        stream = self()
        for batch in stream:
            yield batch