from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchtoolkit.data import create_subsets
from tqdm import tqdm

from models.gpt.gpt2 import GPT
from models.vanilla.transformer import Transformer
from data.dataloader import MIDIDataset
from config.model_config import gpt_model_config, vanilla_model_config
from data.utils import decode_midi
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/taehyeon/data/MAESTRO_dataset', help="Directory to Dataset.")
    parser.add_argument('--model_name', default='GPT', help="Model type.")
    parser.add_argument('--cpt_dir', default='cpt/', help="Path to the checkpoint files.")
    args = parser.parse_args()
    return args

def generate(model, dataloader_test):
    (gen_results_path := Path('gen_res')).mkdir(parents=True, exist_ok=True)
    
    count = 0
    with tqdm(dataloader_test, unit='step') as tstep:
        for batch_idx, batch in enumerate(tstep):
            results = model.generate(batch['input_ids'].to(device))

            for orig, result in zip(batch['input_ids'], results):
                input = orig.cpu().numpy().tolist()
                output = result[len(input):].cpu().numpy().tolist()
                decode_midi(input, file_path=str(gen_results_path) + '/' + f'primer_{count}.mid')
                decode_midi(output, file_path=str(gen_results_path) + '/' + f'{count}.mid')
                count += 1
    
if __name__ == "__main__":  
    args = parse_arguments()
    dataset = MIDIDataset(root_dir=args.data_dir)
    subset_train, subset_valid = create_subsets(dataset, [0.3])
    valid_loader = DataLoader(subset_valid, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    
    # Creates model
    state_dict = torch.load('cpt/Vanillar/1000.ckpt')
    
    # Creates model
    if args.model_name == 'GPT':
        model = GPT(gpt_model_config).to(device)
    elif args.model_name == 'Vanillar':
        model = Transformer(vanilla_model_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    generate(model=model, 
             dataloader_test=valid_loader
             )