# Different inference commands
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
