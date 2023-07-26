import torch
import torch.nn as nn

class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = Multi_Head_Attention(config)
        self.mlp = Feedforward(config)

    def forward(self, x, mask=False):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        
        return x

class Multi_Head_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.d_model // self.n_head

        self.fc_q = nn.Linear(self.d_model, self.d_model)
        self.fc_k = nn.Linear(self.d_model, self.d_model)
        self.fc_v = nn.Linear(self.d_model, self.d_model)
        self.fc_o = nn.Linear(self.d_model, self.d_model)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.scale = torch.sqrt(torch.Tensor([self.head_dim])).to('cuda')

    #Scale Dot-Product Attention
    def Scaled_Dot_Product_Attn(self, Q, K, V, mask):
        score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if mask:
            mask = self.masked_attn_mask(len=score.size(2))
            score = score.masked_fill(mask[:,:,:score.size(2),:score.size(2)], float('-inf'))
        score = torch.softmax(score, dim=-1)

        x = torch.matmul(self.attn_drop(score), V)

        return x
            
    def merge_heads(self, x, shape):
        x = x.contiguous().view(shape)

        return x

    def split_heads(self, x, shape):
        Batch, Time, Channel = shape
        x = x.view(Batch, self.n_head, Time, Channel // self.n_head)
        
        return x
                
    def forward(self, x, mask=False):
        shape = x.size()
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)

        Q, K, V = map(lambda arr: self.split_heads(arr, shape), [Q, K, V])
        
        x = self.Scaled_Dot_Product_Attn(Q, K, V, mask)

        x = self.merge_heads(x, shape=shape)
        x = self.resid_drop(self.fc_o(x))

        return x
    
    def masked_attn_mask(self, len):
        ones = torch.ones(len, len)
        mask = torch.triu(ones, diagonal=1).bool().view(1, 1, len, len)
        return mask.to('cuda')

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

