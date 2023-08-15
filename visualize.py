from musicaiz import loaders, plotters
from musicaiz.plotters import Pianoroll, PianorollHTML
from pathlib import Path
import os

def plot(root, save, file_name):
    midi = loaders.Musa(root + file_name)
    plot = Pianoroll(midi)
    plot.plot_instruments(
        program=[1],
        bar_start=0,
        bar_end=4,
        print_measure_data=False,
        show_bar_labels=False,
        show_grid=False,
        show=True,
        save=True,
        save_path=save+ file_name.split('.')[0]
    )

if __name__ == "__main__":
    track_number = 0
    save_folder='./img/'

    for (root, dirs, files) in os.walk('./gen_res/'):
        for file in files:
            plot(root=root, save=save_folder, file_name=file)

    

