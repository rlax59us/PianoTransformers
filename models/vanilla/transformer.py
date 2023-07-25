"""
This code is written when I took NLP class(2022 Spring) in GIST.
Therefore, it has detaily different from other models' configurations for hyperparameter settings.
"""

import torch
import torch.nn as nn
from models.vanilla.sub_layers import Encoder, Decoder
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self, num_token, max_seq_len=20, dim_model=512, d_k=64, d_v=64, n_head=8, dim_hidden=2048, d_prob=0.1, n_enc_layer=6, n_dec_layer=6):
        super(Transformer, self).__init__()
        self.num_token = num_token
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(num_token, dim_model, n_enc_layer, n_head, dim_hidden, d_prob, device, max_seq_len).to(device)
        self.decoder = Decoder(num_token, dim_model, n_dec_layer, n_head, dim_hidden, d_prob, device, max_seq_len).to(device)
        self.linear = nn.Linear(dim_model, num_token)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, tgt, teacher_forcing=True):
        src_mask = self.pad_mask(src, src)
        src_tgt_mask = self.pad_mask(tgt, src)
        tgt_mask = self.pad_mask(tgt, tgt) * self.masked_attn_mask(tgt, tgt)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, tgt_mask, src_tgt_mask)
        output = self.softmax(self.linear(output))

        return output

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)

    def pad_mask(self, query, key):
        new_key = key.ne(2).unsqueeze(1).unsqueeze(2).repeat(1, 1, query.size(1), 1)
        new_query = query.ne(2).unsqueeze(1).unsqueeze(3).repeat(1, 1, 1, key.size(1))
        padmask = new_key & new_query

        return padmask

    def masked_attn_mask(self, query, key):
        ones = torch.ones(query.size(1), key.size(1))
        mask = torch.tril(ones).bool()
        attnmask = mask.to(device)

        return attnmask