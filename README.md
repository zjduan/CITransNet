## Introduction

This is the pytorch implementation of our paper: A Context-Integrated Transformer-Based Neural Network for Auction Design.  


## Requirements

````
Python>=3.6
Pytorch
Tqdm
````

## Usage

### Generate the data

```bash
cd data_gen
# For Setting G,H,I
python3 data_gen_continous.py

# For Setting D,E,F
python3 data_gen_discrete.py
```

We have already include the data for Setting G in `data_multi/10d_2x5.zip`.

### Train CITransNet

```bash
cd CITransNet
# For Setting G
python3 main_2x5_c.py

# For Setting H
python3 main_3x10_c.py

# For Setting I
python3 main_5x10_c.py

# For Setting D
python3 main_2x5_d.py

# For Setting E
python3 main_3x10_d.py

# For Setting F
python3 main_5x10_d.py
```

## Acknowledgement

Our code is built upon the implementation of https://arxiv.org/abs/2003.01497