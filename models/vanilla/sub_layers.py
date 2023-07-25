import torch
import torch.nn as nn

#Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        index = torch.arange(0, d_model, step=2).float()
        pos_index = pos / (10000**(index/d_model))

        self.encoding[:, 0::2] = torch.sin(pos_index)
        self.encoding[:, 1::2] = torch.cos(pos_index)

    def forward(self, x):
        encod = self.encoding[:x.shape[1], :]

        return encod

#Multi Head Attention
class Multi_Head_Attention(nn.Module):
	def __init__(self, d_model, n_heads, dropout_ratio, device):
		super().__init__()

		self.d_model = d_model
		self.n_heads = n_heads
		self.head_dim = self.d_model // self.n_heads

		self.fc_q = nn.Linear(d_model, d_model)
		self.fc_k = nn.Linear(d_model, d_model)
		self.fc_v = nn.Linear(d_model, d_model)
		self.fc_o = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout_ratio)
		self.scale = torch.sqrt(torch.Tensor([self.head_dim])).to(device)

	#Scale Dot-Product Attention
	def Scaled_Dot_Product_Attn(self, Q, K, V, mask):
		energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
		mask = mask.repeat(1, energy.size(1), 1, 1)
		energy = energy.masked_fill(mask == 0, -1e10)

		attention = torch.softmax(energy, dim=1)

		x = torch.matmul(self.dropout(attention), V)

		return x, attention

	def forward(self, query, key, value, mask=None):
		batch_size = query.shape[0]

		Q = self.fc_q(query)
		K = self.fc_k(key)
		V = self.fc_v(value)

		Q, K, V = map(lambda arr: arr.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3), [Q, K, V])

		x, attention = self.Scaled_Dot_Product_Attn(Q, K, V, mask=mask)

		x = x.permute(0,2,1,3).contiguous()
		x = x.view(batch_size, -1, self.d_model)

		x = self.fc_o(x)

		return x

#Feed Forward Layer
class Feed_Forward(nn.Module):
	def __init__(self, d_model, d_hidden, dropout_ratio):
		super().__init__()

		self.frw = nn.Sequential(
			nn.Linear(d_model, d_hidden),
			nn.ReLU(),
			nn.Dropout(dropout_ratio),
			nn.Linear(d_hidden, d_model)
		)

	def forward(self, x):
		x = self.frw(x)

		return x

class EncoderLayer(nn.Module):
	def __init__(self, d_model, n_heads, d_hidden, dropout_ratio, device):
		super().__init__()

		self.self_attention = Multi_Head_Attention(d_model, n_heads, dropout_ratio, device)
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.feed_forward = Feed_Forward(d_model, d_hidden, dropout_ratio)
		self.layer_norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout_ratio)

	def forward(self, src, src_mask):
		attn_output = self.dropout(self.self_attention(src, src, src, src_mask))
		norm_output = self.layer_norm1(src + attn_output)

		ff_output = self.dropout(self.feed_forward(norm_output))
		final_output = self.layer_norm2(norm_output + ff_output)

		return final_output

#Encoder
class Encoder(nn.Module):
	def __init__(self, input_dim, d_model, n_layers, n_heads, d_hidden, dropout_ratio, device, max_len):
		super().__init__()

		self.device = device

		self.tok_embedding = nn.Embedding(input_dim, d_model)
		self.pos_encoding = PositionalEncoding(d_model, max_len)

		self.layers = nn.ModuleList()
		for l in range(n_layers):
			self.layers.append(EncoderLayer(d_model, n_heads, d_hidden, dropout_ratio, device))

		self.dropout = nn.Dropout(dropout_ratio).to(device)

	def forward(self, src, src_mask):
		batch_size = src.shape[0]
		src_len = src.shape[1]

		pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
		tok_emb = self.tok_embedding(pos).to(self.device)
		pos_enc = self.pos_encoding(pos).to(self.device)
		src = self.dropout(tok_emb + pos_enc)

		for layers in self.layers:
			src = layers(src, src_mask)

		return src

class DecoderLayer(nn.Module):
	def __init__(self, d_model, n_heads, d_hidden, dropout_ratio, device):
		super().__init__()

		self.self_attention = Multi_Head_Attention(d_model, n_heads, dropout_ratio, device)
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.enc_attention = Multi_Head_Attention(d_model, n_heads, dropout_ratio, device)
		self.layer_norm2 = nn.LayerNorm(d_model)
		self.feed_forward = Feed_Forward(d_model, d_hidden, dropout_ratio)
		self.layer_norm3 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout_ratio)

	def forward(self, tgt, enc_src, tgt_mask, src_mask):
		attn_output1 = self.dropout(self.self_attention(tgt, tgt, tgt, tgt_mask))
		norm_output1 = self.layer_norm1(tgt + attn_output1)

		attn_output2 = self.dropout(self.enc_attention(norm_output1, enc_src, enc_src, src_mask))
		norm_output2 = self.layer_norm2(norm_output1 + attn_output2)

		ff_output = self.dropout(self.feed_forward(norm_output2))
		final_output = self.layer_norm3(norm_output2 + ff_output)

		return final_output

#Decoder
class Decoder(nn.Module):
	def __init__(self, input_dim, d_model, n_layers, n_heads, d_hidden, dropout_ratio, device, max_len):
		super().__init__()

		self.device = device

		self.tok_embedding = nn.Embedding(input_dim, d_model)
		self.pos_encoding = PositionalEncoding(d_model, max_len)

		self.layers = nn.ModuleList()
		for l in range(n_layers):
			self.layers.append(DecoderLayer(d_model, n_heads, d_hidden, dropout_ratio, device))

		self.dropout = nn.Dropout(dropout_ratio)

	def forward(self, tgt, enc_src, tgt_mask, src_mask):
		batch_size = tgt.shape[0]
		tgt_len = tgt.shape[1]

		pos = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
		tok_emb = self.tok_embedding(pos).to(self.device)
		pos_enc = self.pos_encoding(pos).to(self.device)
		tgt = self.dropout(tok_emb + pos_enc)

		for layer in self.layers:
			tgt = layer(tgt, enc_src, tgt_mask, src_mask)

		return tgt
