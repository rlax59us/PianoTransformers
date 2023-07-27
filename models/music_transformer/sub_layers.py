import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class MusicTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = Relative_Position_based_Attention(config)
        self.mlp = Feedforward(config)

    def forward(self, x, mask=True):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        
        return x

class Relative_Position_based_Attention(nn.Module):
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
	
		self.max_seq_len = config.n_positions
		self.E = torch.randn([self.n_head, config.n_positions, self.head_dim]).to(device)

	def Relative_Attn(self, Q, K, V, mask=False):
		len_k = K.size(-2)
		len_q = Q.size(-2)

		E = self.E[:, max(0,self.max_seq_len - len_q):,:]
		QE = torch.matmul(Q, E.transpose(-1,-2))
		QE = self.mask_relative_positions(QE)

		S_rel = self.skewing(QE)

		score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
		if mask:
			mask = self.masked_attn_mask(len=score.size(2))
			score = score.masked_fill(mask[:,:,:score.size(2),:score.size(2)], float('-inf'))

		score = score + S_rel
		
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
			
		x = self.Relative_Attn(Q, K, V, mask)

		x = self.merge_heads(x, shape=shape)
		x = self.resid_drop(self.fc_o(x))

		return x
		
	def masked_attn_mask(self, len):
		ones = torch.ones(len, len)
		mask = torch.triu(ones, diagonal=1).bool().view(1, 1, len, len)
		return mask.to('cuda')

	def mask_relative_positions(self, x):
		len = x.shape[-1]
		ones = torch.ones(len, len)
		mask = torch.triu(ones, diagonal=1).flip(1).to(device)
		return x.masked_fill((mask==1), 0)
	
	def skewing(self, x):
		padded_x = F.pad(x, [1,0])
		s = padded_x.shape
		viewed_x = padded_x.view(s[0], s[1], s[3], s[2])

		#take out first (padded) row
		return viewed_x[:,:,1:,:]

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




