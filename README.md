# PianoTransformers
![primer_0](https://github.com/rlax59us/PianoTransformers/assets/56262962/26705fa5-8377-433f-9439-42a3ad388e08)
Simple implementation of Piano Music Generation with several Transformer architectures using Midi Tokenization.

* [Vanilla](https://arxiv.org/abs/1706.03762) ([code](https://github.com/rlax59us/PianoTransformers/tree/main/models/vanilla))
* [GPT](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf) ([code](https://github.com/rlax59us/PianoTransformers/tree/main/models/gpt))
* [Music Transformer](https://arxiv.org/abs/1809.04281) (Google Brain, 2018) ([code](https://github.com/rlax59us/PianoTransformers/tree/main/models/music_transformer))
* [Museformer](https://arxiv.org/abs/2210.10349) (Microsoft Research, 2022) ([code]())

(Will continue to study, analyze, improve, and update!)

## Data
We used [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) which is consist of about 200 hours of piano music.

## Install 
By using Anaconda Environment,
```
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

## Usage

### Train
```
$ python train.py
```
### Generate
```
$ python generate.py
```
## Results
Generated results will be located in gen_res folder and the Visualized Image will be saved in img folder.

You can choose sample what you satisfy among the great number of results.
