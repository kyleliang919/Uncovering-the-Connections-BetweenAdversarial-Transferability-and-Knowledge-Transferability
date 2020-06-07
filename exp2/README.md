# Experiment 2
Adversarial transferability and knowledge transferability among attributes.

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

### Data Zoo 
Download and unzip the data and specify the path in config.py with arguments.

|Database|Version|\#Identity|\#Image|\#Frame|\#Video|Download Link|
|:---:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Raw|5,749|13,233|-|-|[Google Drive](https://drive.google.com/file/d/1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1VzSI_xqiBw-uHKyRbi6zzw)|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Align_250x250|5,749|13,233|-|-|[Google Drive](https://drive.google.com/file/d/11h-QIrhuszY3PzT17Q5eXw8yrewgqX7m/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1Ir8kAcQjBJA6A_pWPL9ozQ)|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Align_112x112|5,749|13,233|-|-|[Google Drive](https://drive.google.com/file/d/1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1Ew5JZ266bkg00jB5ICt78g)|
|[CALFW](https://arxiv.org/pdf/1708.08197.pdf)|Raw|4,025|12,174|-|-|[Google Drive](https://drive.google.com/file/d/1LcIDIfeZ027tbyUJDbaDt12ZoMVJuoMp/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/17IzL_nGzedup1gcPuob0NQ)|
|[CALFW](https://arxiv.org/pdf/1708.08197.pdf)|Align_112x112|4,025|12,174|-|-|[Google Drive](https://drive.google.com/file/d/1kpmcDeDmPqUcI5uX0MCBzpP_8oQVojzW/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1IxqyLFfHNQaj3ibjc7Vcvg)|
|[CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)|Raw|3,884|11,652|-|-|[Google Drive](https://drive.google.com/file/d/1WipxZ1QXs_Fi6Y5qEFDayEgos3rHDRnS/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1gJuZZcm-2crTrqKI0sa5sA)|
|[CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)|Align_112x112|3,884|11,652|-|-|[Google Drive](https://drive.google.com/file/d/14vPvDngGzsc94pQ4nRNfuBTxdv7YVn2Q/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1uqK2LAEE91HYqllgsWcj9A)|
|[Vggface2](https://arxiv.org/pdf/1710.08092.pdf)|Clean|8,631|3,086,894|-|-|[Google Drive](https://drive.google.com/file/d/1jdZw6ZmB7JRK6RS6QP3YEr2sufJ5ibtO/view?usp=sharing)|
|[Vggface2_FP](https://arxiv.org/pdf/1710.08092.pdf)|Align_112x112|-|-|-|-|[Google Drive](https://drive.google.com/file/d/1N7QEEQZPJ2s5Hs34urjseFwIoPVSmn4r/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1STSgORPyRT-eyk5seUTcRA)|
|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|Raw|570|16,488|-|-|[Google Drive](https://drive.google.com/file/d/1FoZDyzTrs8r_oFM3Xqmi3iAHsnoirTRA/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1-E_hkW-bXsXNYRiAhRPM7A)|
|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|Align_112x112|570|16,488|-|-|[Google Drive](https://drive.google.com/file/d/1AoZrZfym5ZhdTyKSxD0qxa7Xrp2Q1ftp/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1ehwmQ4M7WpLylV83uUBxiA)|
|[CFP](http://www.cfpw.io/paper.pdf)|Raw|500|7,000|-|-|[Google Drive](https://drive.google.com/file/d/1tGNtqzWeUx3BYAxRHBbH1Wy7AmyFtZkU/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/10Qq64LO_RWKD2cr_D32_6A)|
|[CFP](http://www.cfpw.io/paper.pdf)|Align_112x112|500|7,000|-|-|[Google Drive](https://drive.google.com/file/d/1-sDn79lTegXRNhFuRnIRsgdU88cBfW6V/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1DpudKyw_XN1Y491n1f-DtA)|
|[CelebA](https://arxiv.org/pdf/1411.7766.pdf)|Raw|10,177|202,599|-|-|[Google Drive](https://drive.google.com/file/d/1FO_p759JtKOf3qOnxOGpmoxCcnKiPdBI/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1DfvDKKEB11MrZcf7hPjJfw)|

## Training

To train the attributes classifiers, run this command:
```
python3 train_clf.py --attr <ith-attribute> --save-model
```
> 
For example:
```
python3 train_clf.py --attr 0 --save-model 
```

> The above command trains a source binary classifier on attribute 0, and save to "checkpoints/0/"

## Transfer
To perform direct transfer, run this command:
```
python3 transfer.py --backbone-resume-root checkpoints/<ith-attribute>/backbone.pth
```
For example:
```
python3 transfer.py --backbone-resume-root checkpoints/0/backbone.pth
```
> Using the attribute classifiers as a feature extractor to perform facial recognition on 7 facial benchmarks.

## Adversarial Attack and Transfer
To perform adversarial attack and calculate per image adversarial loss, run this command:
```
python3 attack_and_transfer.py
```

> To replicate experiment 2 in the paper, run commands in cmd.sh

## Visualization
```
python3 visualize.py
```
> Code to generate the plot in result

## Pre-trained Models

You can download pretrained models here:

- [exp2](https://drive.google.com/drive/folders/19aDKDgyHJ4EblRppRwEBMh7LmbVAL18p?usp=sharing) 

## Results
![exp2](fig4.png)

## Acknowledgement
This code is adapted from the [public repo](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)