import torch
import torch.nn as nn
import math
import networkx as nx
from GraphEmbedding.ge import Node2Vec as OtherNode2Vec
from GraphEmbedding.ge import DeepWalk
import numpy as np


import sys, os
sys.path.append("../")

import Config


class OtherGraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, use_cuda=Config.USE_CUDA):
        super(OtherGraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        
        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size)) 
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
    
    @staticmethod   
    def one_dataset2matrix(data):
        locations = torch.transpose(data, 0, 1)
        return torch.cdist(locations, locations, p=2)
    
    @staticmethod    
    def one_matrix2graph(matrix):
        A = matrix.cpu().detach().numpy()
        return nx.from_numpy_matrix(A, create_using=nx.DiGraph)
        
    def forward(self, inputs):
        pass

    def create_files(self, inputs, ID=0, train_val='train', EmbName = 'OtherNode2Vec'):
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)  
        inputs = inputs.unsqueeze(1)

        #---
        for i in range(batch_size):
            graph = self.one_matrix2graph(self.one_dataset2matrix(inputs[i, 0]))
            nx.write_edgelist(graph, "test.edgelist", data = ['weight'])
            G = nx.read_edgelist("test.edgelist", 
                              create_using = nx.DiGraph(), 
                              nodetype = None, 
                              data = [('weight', float)]) #read graph

            if EmbName == 'OtherNode2Vec':
              model = OtherNode2Vec(G, walk_length = 12, num_walks = 140)
              model.train(embed_size = 128) # train model
              embeddings = model.get_embeddings() # get embedding vectors

            elif EmbName == 'DeepWalk':
              model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)#init model
              model.train(window_size=5,iter=3)# train model
              embeddings = model.get_embeddings()# get embedding vectors
            
            
            embedded = []
            for key in embeddings.keys():   
                embedded.append(torch.unsqueeze(torch.tensor(embeddings[key]), 1))
            embedded = torch.cat(embedded, 1)
            embedded = torch.transpose(embedded, 0, 1)
            embedded = torch.unsqueeze(embedded, 0)

            np.savez_compressed(f'{EmbName}/{EmbName}_{train_val}_{ID}', emb=embedded, data=inputs[i, 0])
            print(f'{EmbName}/{EmbName}_{train_val}_{ID}')
