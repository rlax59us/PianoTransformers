import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchtoolkit.data import create_subsets

from models.gpt.gpt import GPT
from models.vanilla.transformer import Transformer
from models.music_transformer.music_transformer import MusicTransformer
from data.dataloader import MIDIDataset
from config.training_config import training_config
from config.model_config import *
from itertools import cycle
from tqdm import tqdm
import wandb
import os
import argparse
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/taehyeon/data/MAESTRO_dataset', help="Directory to Dataset.")
    parser.add_argument('--model_name', default='Music', help="Model type.")
    parser.add_argument('--cpt_dir', default='cpt/', help="Path to the checkpoint files.")
    args = parser.parse_args()
    return args

def validating(model, validloader):
    model.eval()
    valid_loss = 0
    N = len(validloader)
    for batch_idx, batch in enumerate(validloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        logits, loss = model.forward(input_ids=input_ids, labels=labels)
        valid_loss += loss.item()
    valid_loss /= N

    return valid_loss

def training(model, trainloader, validloader, optimizer, scheduler, cpt_path="cpt/"):
    # Training
    wandb.init(
        project="MusicGPT"
    )
    
    with tqdm(cycle(trainloader), unit='step') as tstep:
        for batch_idx, batch in enumerate(tstep):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits, loss = model.forward(input_ids=input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            tstep.set_postfix(train_loss=loss.item())
            wandb.log({"train_loss": loss})

            if (batch_idx + 1) % (training_config.eval_steps / training_config.train_batch_size) == 0:
                valid_loss = validating(model=model, validloader=validloader)
                tstep.set_postfix(valid_loss=valid_loss)
                wandb.log({"valid_loss": valid_loss})
                model.train()
                if not os.path.exists(cpt_path):
                    os.makedirs(cpt_path)
                torch.save(model.state_dict(), cpt_path + str((batch_idx+1)*training_config.train_batch_size)+'.ckpt')

            if batch_idx > training_config.max_steps / training_config.train_batch_size:
                break

    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    dataset = MIDIDataset(root_dir=args.data_dir)
    subset_train, subset_valid = create_subsets(dataset, [0.3])
    train_loader = DataLoader(subset_train, batch_size=training_config.train_batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(subset_valid, batch_size=training_config.eval_batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Creates model
    if args.model_name == 'GPT':
        model = GPT(gpt_model_config).to(device)
    elif args.model_name == 'Vanilla':
        model = Transformer(vanilla_model_config).to(device)
    elif args.model_name == 'Music':
        model = MusicTransformer(music_transformer_model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=100, T_mult=1, eta_min=1e-5)

    training(model=model, 
             trainloader=train_loader, 
             validloader=valid_loader, 
             optimizer=optimizer,
             scheduler=scheduler,
             cpt_path=args.cpt_dir + args.model_name + '/'
             )
