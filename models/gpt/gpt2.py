import torch
import torch.nn as nn
from torch.nn import functional as F

from models.gpt.block import GPTBlock
from data.constants import *

device = "cuda" if torch.cuda.is_available() else "cpu"
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad_token = TOKEN_PAD
        self.eos_token = TOKEN_END
        self.vocab_size = config.vocab_size

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.masked_block = GPTBlock(config)
        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.n_layer - 1)])
        # decoder head
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, labels=None):
        # forward the GPT model
        embeddings = self.tok_emb(input_ids) + self.pos_emb[:, :input_ids.size(-1), :]
        x = self.drop(embeddings)
        x = self.masked_block(x, mask=True)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)

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
