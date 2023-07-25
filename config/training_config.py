"""
#GPT
class training_config:
    train_batch_size=16
    eval_batch_size=12
    eval_steps=1000
    learning_rate=1e-4
    weight_decay=0.01
    max_grad_norm=3.0
    max_steps=25000
"""

#Vanillar
class training_config:
    train_batch_size=2
    eval_batch_size=12
    eval_steps=100
    learning_rate=1e-4
    weight_decay=0.01
    max_grad_norm=3.0
    max_steps=2500