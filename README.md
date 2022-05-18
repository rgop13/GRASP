## GRASP: Guiding model with RelAtional Semantics using Prompt
## 0. Package Description
```
├─ config/: model config files for transformers
    ├─ GRASP-base.json
    ├─ GRASP-large.json
├─ data/: datasets
    ├─ datav1/: dialogRE dataset (version 1) 
    ├─ datav2/: dialogRE dataset (version 2) 
    ├─ k-shot/: dialogRE examples for few-shot experiments
        ├─ 8-1/
        ├─ 8-2/
        ├─ 8-3/
        ├─ 16-1/
        ├─ ... 
├─ data_reader/: source codes for data/model processor
    ├─ data_processor.py
    ├─ model_processor.py
├─ models/: source codes for GRASP architecture
    ├─ GRASP.py: GRASP model without manual init. for RCD
    ├─ GRASP_minit.py: GRASP model
├─ utils/
    ├─ bert_utils.py
    ├─ data_utils.py: utils for processing data
    ├─ eval_utils.py
    ├─ general_utils.py
├─ evaluate.py
├─ train.py
├─ train_base.sh
├─ train_large.sh
├─ README.md
```

## 1. Environments
We conducted experiments on a server with a RTX A6000 (48GB) GPU.
- python      (3.6.13)  
- CUDA        (11.1)  
- UbuntuOS 20.04.1 LTS

## 2. Dependencies
- torch                    (1.8.1)
- transformers             (4.7.0)
- tqdm     (4.49.0)
- numpy                    (1.19.5)
- attrdict (2.0.1)
- setproctitle (1.2.2)
- scikit-learn (0.24.2)
- tensorboard (2.7.0)
- tensorboardX (2.4.1)
- sklearn (0.0)

## 3. Training/Evaluating for reproducing
If you want to reproduce our results, please follow our hyper-parameter settings and run the bash file with the following command.

For the case of GRASP_base:
```
# For data v1
CUDA_VISIBLE_DEVICES=0 sh train_base.sh

# For data v2
CUDA_VISIBLE_DEVICES=0 sh train_base_v2.sh
```
For the case of GRASP_large:
```
# For data v1
CUDA_VISIBLE_DEVICES=0 sh train_large.sh

# For data v2
CUDA_VISIBLE_DEVICES=0 sh train_large_v2.sh
```

You can also reproduce our **few-shot results** as below.
```
# 8-shots
CUDA_VISIBLE_DEVICES=0 python train.py --config_file GRASP-base --save_dirpath checkpoint/grasp_base_8shot1 --data_dir data/k-shot/8-1 --mode train
CUDA_VISIBLE_DEVICES=0 python train.py --config_file GRASP-base --save_dirpath checkpoint/grasp_base_8shot1 --data_dir data/k-shot/8-1 --mode evaluation
CUDA_VISIBLE_DEVICES=0 python train.py --config_file GRASP-base --save_dirpath checkpoint/grasp_base_8shot2 --data_dir data/k-shot/8-2 --mode train
CUDA_VISIBLE_DEVICES=0 python train.py --config_file GRASP-base --save_dirpath checkpoint/grasp_base_8shot2 --data_dir data/k-shot/8-2 --mode evaluation
CUDA_VISIBLE_DEVICES=0 python train.py --config_file GRASP-base --save_dirpath checkpoint/grasp_base_8shot3 --data_dir data/k-shot/8-3 --mode train
CUDA_VISIBLE_DEVICES=0 python train.py --config_file GRASP-base --save_dirpath checkpoint/grasp_base_8shot3 --data_dir data/k-shot/8-3 --mode evaluation
...

# 16-shots
CUDA_VISIBLE_DEVICES=0 python train.py --config_file GRASP-base --save_dirpath checkpoint/grasp_base_16shot1 --data_dir data/k-shot/16-1 --mode train
CUDA_VISIBLE_DEVICES=0 python train.py --config_file GRASP-base --save_dirpath checkpoint/grasp_base_16shot1 --data_dir data/k-shot/16-1 --mode evaluation
...
```
