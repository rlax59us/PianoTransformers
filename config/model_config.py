from transformers import GPT2Config

model_config = GPT2Config(
        vocab_size=1024,
        n_positions=1024,
        n_embd=512,
        n_layer=8,
        n_head=8,
        n_inner=1024,
        resid_pdrop=.1,
        embd_pdrop=.1,
        attn_pdrop=.1
    )