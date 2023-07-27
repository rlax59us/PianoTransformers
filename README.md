# PianoTransformers
Simple implementation of Piano Music Generation with several Transformer architectures using Midi Tokenization.

* Vanilla ([code](https://github.com/rlax59us/PianoTransformers/tree/main/models/vanilla))
* GPT ([code](https://github.com/rlax59us/PianoTransformers/tree/main/models/gpt))
* [Music Transformer](https://arxiv.org/abs/1809.04281) ([code](https://github.com/rlax59us/PianoTransformers/tree/main/models/music_transformer))
* [Museformer](https://arxiv.org/abs/2210.10349) ([code]())

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
Generated results are located in gen_res folder.

You can choose sample what you satisfy among the great number of results.