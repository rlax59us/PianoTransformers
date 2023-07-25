class gpt_model_config:
    vocab_size=1024
    n_positions=1024
    n_embd=512
    n_layer=8
    n_head=8
    n_inner=1024
    resid_pdrop=.1
    embd_pdrop=.1
    attn_pdrop=.1

class vanilla_model_config:
    num_token=1024
    max_seq_len=1024
    dim_model=256
    d_k=64
    d_v=64
    n_head=4
    dim_hidden=1024
    d_prob=0.1
    n_enc_layer=3
    n_dec_layer=3

class t5_model_config:
    pass