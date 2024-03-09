# $\rm{[MindSpore-phase3]}$ $DNF$

本项目包含了以下论文的mindspore实现：

> **DNF: Decouple and Feedback Network for Seeing in the Dark**
>
> Xin Jin Ling-Hao Han Zhen Li Chun-Le Guo1 Zhi Chai Chongyi Li
> 
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2023

[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.pdf)]



文章官方版本仓库链接: https://github.com/Srameo/DNF


目前已完成完整代码的mindspore实现


## requirements

python 3.8

mindspore: 2.2.11

cuda: 11.1

## Train and Evaluation
Due to the large size of the training data, please refer to the PyTorch version for the exact steps.

> https://github.com/Srameo/DNF/blob/main/docs/benchmark.md

Then run the code as followed to train with your own config

> python runner.py -cfg [CFG]
