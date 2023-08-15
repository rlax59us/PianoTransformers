import torch
import torch.nn as nn

class MuseformerDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_regular_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        #self.summary_tokens = nn.Embedding(config., config.n_embd)

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = Multi_Head_Attention(config)
        self.mlp = Feedforward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        
        return x

class Multi_Head_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

class Fine_Grained_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

class Coarse_Grained_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

class Feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.n_embd, config.n_inner)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.dropout(self.linear2(x))

        return x

