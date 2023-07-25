# PianoTransformers
Simple implementation of Piano Music Generation with several Transformer(Vanilla, GPT, T5) architectures using Midi Tokenization.

(Will continue to study, analyze, improve, and update!)

## Background
GPT models are a type of Neural Network that consists of a couple of decoder blocks.

We used [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) which is consist of about 200 hours of piano music.

In this practice, by applying GPT2 and tokenization of MIDI format, the Music Generation problem will be considered a token sequence generation and solved.

## Prerequisite
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