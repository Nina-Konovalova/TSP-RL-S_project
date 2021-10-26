from train_RL.CombinatorialRL import CombinatorialRL
from train_RL.CombinatorialRL import reward
from train_RL.train_model import TrainModel
import Config
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from arguments import parse_args
from dataset.TSPDataset import TSPDataset


import warnings
warnings.simplefilter("ignore")


def setup_experiment(title, logdir="./logs"):
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, experiment_name))
    best_model_path = f"{title}.best.pth"
    return writer, experiment_name, best_model_path




def main():
    args = parse_args()

    tsp_20_model = CombinatorialRL(
        args.embedding_size,
        Config.HIDDEN_SIZE,
        20,
        Config.N_GLIMPSE, 
        Config.TANH_EXPLORATION,
        Config.USE_TANH ,
        reward,
        attention="Dot",
        embedding_type = args.embedding_type,
        batch_size=args.batch_size,
        use_cuda=Config.USE_CUDA)

    if Config.USE_CUDA:
        tsp_20_model = tsp_20_model.cuda()
    
    
    writer, experiment_name, best_model_path = setup_experiment('PointNet', logdir="./logs")
    print(f"Experiment name: {experiment_name}")

    if args.embedding_type == 'simple' or args.embedding_type == 'linear':
        train_size = args.train_size
        val_size = args.val_size
        train_dataset = TSPDataset(20, train_size)
        val_dataset   = TSPDataset(20, val_size)
    else:
        import pandas as pd
        df_train = pd.read_csv(args.path_train)
        df_val = pd.read_csv(args.path_val)
        train_dataset = list(zip(df_train.path_train))
        val_dataset = list(zip(df_val.path_val))


    tsp_20_train = TrainModel(tsp_20_model, 
                        train_dataset, 
                        val_dataset, 
                        best_model_path,
                        writer,
                        args.embedding_type,
                        threshold=3.99)

    tsp_20_train.train_and_validate(args.epochs)



if __name__ == '__main__':
    main()
