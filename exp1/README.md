# Experiment 1
Adversarial transferability and knowledge transferability among data distributions.

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Training

To train the source/reference model(s) on different classes on different datasets, run this command:
```
python3 train_clf.py -d <dataset> -a <architecture> --include_list <classes_included> --save_name <save_path>
```

For example:
```
python3 train_clf.py -d cifar10 -a res_net18 --include_list 0 1 8 9 --save_name 0_percent.pth 
```

> The above command trains a source Resnet18 on CIFAR-10 class 0,1,8,9 and save to "checkpoints/0_percent.pth"

## Transfer
To perform direct transfer/fine tuning (w/o transfer_last flag), run this command:
```
python3 transfer.py -d <dataset> -a <architecture> --include_list <class_included> --load_name <source_path > --transfer_last
```
For example:
```
python3 transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5 --load_name 0_percent.pth --transfer_last
```
> This performs direct transfer from the previously trained model CIFAR-10 model to STL-10. Learning loss/acc and model will be recorded.

## Adversarial Attack and Transfer
To perform adversarial attack and calculate per image adversarial loss, run this command:
```
python3 attack_and_transfer.py -d stl10 -a res_net18 --include_list 1 3 4 5
```

> To replicate experiment 1 in the paper, run commands in cmd.sh

## Visualization
```
python3 visualize.py
```
> Code to generate the two graphs in result

## Pre-trained Models

You can download pretrained models here:

- [exp1](https://drive.google.com/drive/folders/1ftVLd7zwZ2Pfr3pm2sPEV1bCdkmjKCzo?usp=sharing) 

## Results
![exp1](fig3.png)

## Acknowledgement
This code is adapted from [official pytorch tutorial](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py)