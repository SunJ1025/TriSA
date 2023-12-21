# TriSSA: A Three Stage Strong Approach for UAV-Satellite Cross-View Geo-Localization Based on Self-Supervised Feature Enhancement

![Python 3.6+](https://img.shields.io/badge/README-ENGLISH-green.svg)


This repository is the code for our paper [TriSSA: A Three Stage Strong Approach for UAV-Satellite Cross-View Geo-Localization Based on Self-Supervised Feature Enhancement](), Thank you for your kindly attention.

## requirement
1. Download the [University-1652](https://github.com/layumi/University1652-Baseline) dataset
2. Prepare Data Folder 
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

## Evaluation and Get the Results in Our Paper
You can download the trained embedding files (.mat)from the following link and put them in the **"evaluaiton/weights/"** folder

### Download the trained files
[Google Driver](https://drive.google.com/drive/folders/1rl5wZCL3-WdbB7lSOQld_L-TKET8Td3p?usp=drive_link)


You can download the trained embedding files (.mat) from the following link and put them in the **"evaluaiton/weights/"** folder

We prepared the following **.py** files for easy evaluation:

```
├── evaluation/
│   ├── evaluate_cpu_no_rerank.py  /* test with cpu and no rerank
│   ├── evaluate_cpu_rerank.py     /* test with cpu and rerank
│   ├── evaluate_gpu_rerank.py     /* test with gpu and rerank     
```

If you are using the gpu-based re-ranking, make sure to compile the file by:
```
cd evaluation/
sh make.sh
```
### Citation
```bibtex
@article{zhang2020understanding,
  title={Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective},
  author={Xuanmeng Zhang, Minyue Jiang, Zhedong Zheng, Xiao Tan, Errui Ding, Yi Yang},
  journal={arXiv preprint arXiv:2012.07620},
  year={2020}
}
```

## Train and Test
We provide scripts to complete FSRA training and testing
* Change the **data_dir** and **test_dir** paths and then run:
```shell
python train.py --gpu_ids 0 --name traied_model_name --train_all --batchsize 32  --data_dir your_data_path
```

```shell
python test.py --gpu_ids 0 --name traied_model_name --test_dir your_data_path  --batchsize 32 --which_epoch 120
```