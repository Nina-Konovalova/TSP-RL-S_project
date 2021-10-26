# RL-Travelling Salesman Problem :robot:
----------------------------------

Our project is based on Reinforcement Learning (RL) for solving Travelling Salesman Problem (TSP). Our code and experiments around the paper https://arxiv.org/abs/1802.04240.

We consider solving TCP solving with RL based on [Pointer Network](https://arxiv.org/abs/1506.03134). 

<p align="center">
  <img src="https://github.com/Nina-Konovalova/TCP-RL-Skoltech_project/blob/main/pictures/data/image.png" width="430" height="250">
</p>

As the dataset 20 uniform distributed points from 0 to 1 for each coordinates were used. 

<p align="center">
  <img src="https://github.com/Nina-Konovalova/TCP-RL-Skoltech_project/blob/main/pictures/data/map.png" >
</p>

For each dataset 4 different types of embeddings were used:

:lock: Linear layer;

:lock: Simple node encoding throw dot product with a random vector;

:lock: [Node2Vec](https://arxiv.org/abs/1607.00653);

:lock: [DeepWalk](https://paperswithcode.com/method/deepwalk).


## Quick start
-----------------------------------
To train for 30 epochs and infer dataset containing 20 points, using simple embeddings with embedding size 128 you may just run:

```python main_not.py```


### Different inference commands
--------------------------------------

### model

`-e` or `--epochs` - number of epochs. Default: **30**;

`-embedding` or `--embedding_size` - size of embeddings. Default: **128**;

`--embedding_type` - type of embeddings. Default: **simple**. Other possible options: **linear**, **other**;

`-b` or `--batch_size` - batch size. Default: **1024**;


### dataset

`-train_size` - size of train dataset for linear and simple embeddings. Default: **100_000**;

`--val_size` - size of val dataset for linear and simple embeddings. Default: **1_000**;

`--path_train` - path for other saved train embedding. Default: **OtherNode2Vec_train.csv**;

`--path_val` - path for other saved val embeddings. Default: **OtherNode2Vec_val.csv**.


## Your embeddings
If you want to use other embeddings, for each item of dataset you should save **.npz** file, that contains:

- embeddings;
- initial item;

Then you should make a **.csv** files for *train* and *val* datasets, that constains 

## Results
-----------------------------------

<p align="center">
  <img src="https://github.com/Nina-Konovalova/TCP-RL-Skoltech_project/blob/main/pictures/results/map_1.png" >
  <img src="https://github.com/Nina-Konovalova/TCP-RL-Skoltech_project/blob/main/pictures/results/map_2.png" >
</p>


#### Train and val losses for linear and simple embeddings.

- Pink color for linear embeddings;
- Blue color for simple embeddings;

<p align="center">
  <img src="https://github.com/Nina-Konovalova/TCP-RL-Skoltech_project/blob/main/pictures/results/train_lin_s.jpg", alt="Train loss for linear and simple embeddings" >
  <img src="https://github.com/Nina-Konovalova/TCP-RL-Skoltech_project/blob/main/pictures/results/val_lin_s.jpg", alt="Val loss for linear and simple embeddings" >
</p>

