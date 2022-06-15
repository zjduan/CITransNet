## Introduction

This is the Pytorch implementation of our paper: *A Context-Integrated Transformer-Based Neural Network for Auction Design* ([https://arxiv.org/abs/2201.12489]()) in *ICML 2022*.


## Requirements


* Python >= 3.6
* Pytorch 1.10.0
* Argparse
* Logging
* Tqdm
* Scipy

## Usage

### Generate the data

```bash
cd data_gen
# For Setting G,H,I
python data_gen_continous.py

# For Setting D,E,F
python data_gen_discrete.py
```

### Train CITransNet

```bash
cd CITransNet
# For Setting G
python main_2x5_c.py

# For Setting H
python main_3x10_c.py

# For Setting I
python main_5x10_c.py

# For Setting D
python main_2x5_d.py

# For Setting E
python main_3x10_d.py

# For Setting F, it is recommended to train it with 2 GPUs.
CUDA_VISIBLE_DEVICES=0,1 python main_5x10_d.py --data_parallel True
```

## Citation
If you find our code useful in your research, please cite the original paper.
```
@article{duan2022context,
	title={A Context-Integrated Transformer-Based Neural Network for Auction Design},  
	author={Duan, Zhijian and Tang, Jingwu and Yin, Yutong and Feng, Zhe and Yan, Xiang and Zaheer, Manzil and Deng, Xiaotie},  
	journal={arXiv preprint arXiv:2201.12489},  
	year={2022}  
}
```


## Acknowledgement

Our code is built upon the implementation of [https://arxiv.org/abs/2003.01497]()