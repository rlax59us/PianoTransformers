"""
The original code is written when I(rlax59us) took NLP class(2022 Spring) in GIST.
Therefore, it might be detaily different from other models' configurations for hyperparameter settings.
"""

import torch
import torch.nn as nn
from models.vanilla.sub_layers import Encoder, Decoder
from torch.nn import functional as F
from data.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.pad_token = TOKEN_PAD
        self.eos_token = TOKEN_END
        self.num_token = config.num_token
        self.max_seq_len = config.max_seq_len
        self.encoder = Encoder(config).to(device)
        self.decoder = Decoder(config).to(device)
        self.linear = nn.Linear(config.dim_model, config.num_token)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, labels=None):
        enc_src = self.encoder(input_ids)
        if labels == None:
            logits = self.decoder(input_ids, enc_src)
        else:
            logits = self.decoder(labels, enc_src)
        logits = self.linear(logits)
        
        loss = None

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss

    def generate(self, primer=None, target_seq_length=1024):
        num_primer = len(primer)
        len_primer = len(primer[0])
        gen_tokens = torch.LongTensor([self.pad_token for i in range(target_seq_length-len_primer)]).expand(num_primer, target_seq_length-len_primer)
        gen_tokens = torch.concat((primer.type(torch.long).to(device), gen_tokens.to(device)), dim=-1).to(device)

        i = num_primer
        while (i < target_seq_length):
            logits, _ = self.forward(gen_tokens[..., :i])
            probs = self.softmax(logits)[..., :self.eos_token]
            token_probs = probs[:, i - 1, :]

            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            gen_tokens[:, i] = next_token

            if next_token == self.eos_token:
                break
            i += 1

        return gen_tokens[:, :i]

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)