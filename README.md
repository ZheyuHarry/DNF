# DNF
This is the Mindspore version of DNF: Decouple and Feedback Network for Seeing in the Dark, CVPR 2023

The official PyTorch implementation, pretrained models and examples are available at https://github.com/Srameo/DNF

## requirements

python 3.8

mindspore: 2.2.11

cuda: 11.1

## Train and Evaluation
Due to the large size of the training data, please refer to the PyTorch version for the exact steps.

> https://github.com/Srameo/DNF/blob/main/docs/benchmark.md

Then run the code as followed to train with your own config

> python runner.py -cfg [CFG]
