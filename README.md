# TriSSA: A Three Stage Strong Approach for UAV-Satellite Cross-View Geo-Localization Based on Self-Supervised Feature Enhancement

![Python 3.6+](https://img.shields.io/badge/README-ENGLISH-green.svg)

## Prerequisites
- Python 3.6
- GPU Memory >= 4G
- Numpy
- Pytorch 0.3+ (http://pytorch.org/)


## Getting started
Check the Prerequisites. The download links for this practice are:

- Data: [University-1652](https://github.com/layumi/University1652-Baseline/blob/master/Request.md)

## Part 1: Training
### Prepare Data Folder 
```
├── University-1652/
│   ├── readme.txt
│   ├── train/
│       ├── drone/                   /* drone-view training images 
│           ├── 0001
|           ├── 0002
|           ...
│       ├── street/                  /* street-view training images 
│       ├── satellite/               /* satellite-view training images       
│       ├── google/                  /* noisy street-view training images (collected from Google Image)
│   ├── test/
│       ├── query_drone/  
│       ├── gallery_drone/  
│       ├── query_street/  
│       ├── gallery_street/ 
│       ├── query_satellite/  
│       ├── gallery_satellite/ 
│       ├── 4K_drone/
```



### Training (`python train.py`)

```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of the model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.



## Part 2: Test
### Part 2.1: Extracting feature (`python test.py`)
In this part, we load the network weight (we just trained) to extract the visual feature of every image.
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--name` the dir name of the trained model.


`--batchsize` batch size.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


### Part 2.2: Evaluation
Yes. Now we have the feature of every image. The only thing we need to do is matching the images by the feature.
```bash
python evaluate_gpu.py
```




