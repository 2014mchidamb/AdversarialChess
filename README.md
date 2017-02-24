# Adversarial Chess

TensorFlow implementation of [Style Transfer Generative Adversarial Networks: Learning to Play Chess Differently](https://openreview.net/pdf?id=HkpbnufYe).

## Requirements

To run this project, working installations of [TensorFlow](https://www.tensorflow.org/install/), [Python-Chess](http://python-chess.readthedocs.io/en/latest/), and [h5py](http://docs.h5py.org/en/latest/quick.html) are needed. TensorFlow version 0.12.1 was used.

## Background

AIs for chess have long since exceeded the abilities of the top human chess players. However, current AIs offer little pedagogical value due to their mechanical playstyle. This research project hopes to overcome this by applying the idea of style transfer to chess, so that an AI can be trained to play in the style of specific human players. 

## Data

The two datasets used in this project can be obtained from [FICS](http://ficsgames.org/download.html) and [PGNMentor](http://www.pgnmentor.com/files.html).

## Training and Testing

The model can be trained with:

```python
python train_model.py
```

And run with:

```python
python play.py
```

## Examples

![Commandline Move 1](/Examples/cmdmove1.png) ![Commandline Move 2](/Examples/cmdmove2.png)

