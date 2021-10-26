import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')

    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of epochs')

    parser.add_argument('-embedding', '--embedding_size', default=128, type=int,
                        help='size of embeddings')

    parser.add_argument('--embedding_type', default='simple', type=str,
                        help='type of embeddings')
    
    parser.add_argument('-b', '--batch_size', default=1024, type=int,
                        help='batch_size')

    # dataset
    parser.add_argument('--train_size', default=100_000, type=int,
                        help='size of train dataset for linear and simple embeddings')
    parser.add_argument('--val_size', default=1_000, type=int,
                        help='size of val dataset for linear and simple embeddings')

    parser.add_argument('--path_train', default='OtherNode2Vec_train.csv', type=str,
                        help='path for saved other train embeddings')
    parser.add_argument('--path_val', default='OtherNode2Vec_val.csv', type=str,
                        help='path for saved other val embeddings')

    args = parser.parse_args()
    return args
