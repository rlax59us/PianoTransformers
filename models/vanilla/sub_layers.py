import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

#Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_seq_len, dim_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_seq_len)
        pos = pos.float().unsqueeze(dim=1)
        index = torch.arange(0, dim_model, step=2).float()
        pos_index = pos / (10000**(index/dim_model))

        self.encoding[:, 0::2] = torch.sin(pos_index)
        self.encoding[:, 1::2] = torch.cos(pos_index)

    def forward(self, x):
        encod = x + self.encoding[:x.shape[1], :].to(device)

        return encod

class Multi_Head_Attention(nn.Module):
	def __init__(self, config):
		super().__init__()
		
		self.dim_model = config.dim_model
		self.n_head = config.n_head
		self.head_dim = self.dim_model // self.n_head

		self.fc_q = nn.Linear(self.dim_model, self.dim_model)
		self.fc_k = nn.Linear(self.dim_model, self.dim_model)
		self.fc_v = nn.Linear(self.dim_model, self.dim_model)
		self.fc_o = nn.Linear(self.dim_model, self.dim_model)
		self.attn_drop = nn.Dropout(config.d_prob)
		self.resid_drop = nn.Dropout(config.d_prob)
		self.scale = torch.sqrt(torch.Tensor([self.head_dim])).to(device)

	#Scale Dot-Product Attention
	def Scaled_Dot_Product_Attn(self, Q, K, V, mask=False):
		score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
		attention = torch.softmax(score, dim=1)
		if mask:
			mask = self.masked_attn_mask(len=score.size(2))
			attention = attention.masked_fill(mask[:,:,:score.size(2),:score.size(2)], float('-inf'))

		x = torch.matmul(self.attn_drop(attention), V)

		return x
				
	def merge_heads(self, x, shape):
		x = x.contiguous().view(shape)

		return x

	def split_heads(self, x, shape):
		Batch, Time, Channel = shape
		x = x.view(Batch, self.n_head, Time, Channel // self.n_head)
			
		return x
					
	def forward(self, query, key, value, mask=False):
		shape = query.size()

		Q = self.fc_q(query)
		K = self.fc_k(key)
		V = self.fc_v(value)

		Q, K, V = map(lambda arr: self.split_heads(arr, shape), [Q, K, V])
			
		x = self.Scaled_Dot_Product_Attn(Q, K, V)

		x = self.merge_heads(x, shape=shape)
		x = self.resid_drop(self.fc_o(x))

		return x
		
	def masked_attn_mask(self, len):
		ones = torch.ones(len, len)
		mask = torch.tril(ones).bool().view(1, 1, len, len)
		return mask.to('cuda')
    
#Feed Forward Layer
class Feed_Forward(nn.Module):
	def __init__(self, dim_model, d_hidden, dropout_ratio):
		super().__init__()

		self.frw = nn.Sequential(
			nn.Linear(dim_model, d_hidden),
			nn.ReLU(),
			nn.Dropout(dropout_ratio),
			nn.Linear(d_hidden, dim_model)
		)

	def forward(self, x):
		x = self.frw(x)

		return x

class EncoderLayer(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.self_attention = Multi_Head_Attention(config)
		#self.self_attention = RelativePositionBasedAttention(config)
		self.layer_norm1 = nn.LayerNorm(config.dim_model)
		self.feed_forward = Feed_Forward(config.dim_model, config.dim_hidden, config.d_prob)
		self.layer_norm2 = nn.LayerNorm(config.dim_model)
		self.dropout = nn.Dropout(config.d_prob)

	def forward(self, src):
		attn_output = self.dropout(self.self_attention(src, src, src))
		norm_output = self.layer_norm1(src + attn_output)

		ff_output = self.dropout(self.feed_forward(norm_output))
		final_output = self.layer_norm2(norm_output + ff_output)

		return final_output

#Encoder
class Encoder(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dim_model = config.dim_model
		self.tok_embedding = nn.Embedding(config.num_token, config.dim_model)
		self.pos_encoding = PositionalEncoding(config.dim_model, config.max_seq_len)

		self.layers = nn.ModuleList()
		for l in range(config.n_enc_layer):
			self.layers.append(EncoderLayer(config))

		self.dropout = nn.Dropout(config.d_prob)

	def forward(self, src):
		src = self.tok_embedding(src)
		src *= math.sqrt(self.dim_model)
		src = self.pos_encoding(src)
		src = self.dropout(src)

		for layers in self.layers:
			src = layers(src)

		return src

class DecoderLayer(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.self_attention = Multi_Head_Attention(config)
		#self.self_attention = RelativePositionBasedAttention(config)
		self.layer_norm1 = nn.LayerNorm(config.dim_model)
		self.cross_attention = Multi_Head_Attention(config)
		#self.cross_attention = RelativePositionBasedAttention(config)
		self.layer_norm2 = nn.LayerNorm(config.dim_model)
		self.feed_forward = Feed_Forward(config.dim_model, config.dim_hidden, config.d_prob)
		self.layer_norm3 = nn.LayerNorm(config.dim_model)
		self.dropout = nn.Dropout(config.d_prob)

	def forward(self, tgt, enc_src):
		attn_output1 = self.dropout(self.self_attention(tgt, tgt, tgt, mask=True))
		norm_output1 = self.layer_norm1(tgt + attn_output1)

		attn_output2 = self.dropout(self.cross_attention(norm_output1, enc_src, enc_src))
		norm_output2 = self.layer_norm2(norm_output1 + attn_output2)

		ff_output = self.dropout(self.feed_forward(norm_output2))
		final_output = self.layer_norm3(norm_output2 + ff_output)

		return final_output

#Decoder
class Decoder(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dim_model = config.dim_model
		self.tok_embedding = nn.Embedding(config.num_token, config.dim_model)
		self.pos_encoding = PositionalEncoding(config.dim_model, config.max_seq_len)

		self.layers = nn.ModuleList()
		for l in range(config.n_dec_layer):
			self.layers.append(DecoderLayer(config))

		self.dropout = nn.Dropout(config.d_prob)

	def forward(self, tgt, enc_src):
		tgt = self.tok_embedding(tgt)
		tgt *= math.sqrt(self.dim_model)
		tgt = self.pos_encoding(tgt)
		tgt = self.dropout(tgt)

		for layer in self.layers:
			tgt = layer(tgt, enc_src)

		return tgt
