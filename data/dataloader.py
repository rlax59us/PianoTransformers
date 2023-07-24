from pathlib import Path
import torch
from torch.utils.data import IterableDataset
from data.utils import encode_midi

import random
from data.constants import *

# process_midi
def process_midi(encoded_midi, max_seq_len):
    encoded_len = len(encoded_midi)

    source = [TOKEN_PAD for i in range(max_seq_len)]
    target = [TOKEN_PAD for i in range(max_seq_len)]

    if encoded_len == 0:
        return source, target
    
    elif encoded_len <= max_seq_len:
        source[:encoded_len] = encoded_midi
        target[:encoded_len-1] = encoded_midi[1:]
        target[encoded_len-1] = TOKEN_END
    else:
        # Randomly selecting a range
        start = random.randint(0, encoded_len - max_seq_len - 1)
        data = list(encoded_midi[start:start+max_seq_len+1])

        source = data[:max_seq_len]
        target = data[1:max_seq_len+1]

    return source, target

class MIDIDataset(IterableDataset):
    def __init__(self, root_dir='/home/taehyeon/data/MAESTRO_dataset'):
        self.midi_paths = list(Path(root_dir).glob('**/*.mid')) + list(Path(root_dir).glob('**/*.midi'))

    def __getitem__(self, idx):
        midi_path = self.midi_paths[idx]

        encoded_midi = encode_midi(str(midi_path))
        source, target = process_midi(encoded_midi, max_seq_len=512)
        return {"input_ids": torch.LongTensor(source), "labels": torch.LongTensor(target)}
    
    def __len__(self): 
        return len(self.midi_paths)