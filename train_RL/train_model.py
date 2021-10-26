import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import math
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append("../")

from model.attention import Attention
import Config
from embeddings import *
from train_RL.render import render


import warnings
warnings.simplefilter("ignore")


def collate_batch(batch):
    embed = []
    input = []
    #path = "DeepWalk/"
    for train_path in batch:
      data = np.load(train_path[0], allow_pickle=True)
      embeddings = data['emb'].squeeze(0)
      initial = data['data']
      embed.append(embeddings)
      input.append(initial)

    return [embed, input]  



class TrainModel:
    def __init__(self, model, train_dataset, val_dataset, best_model_path, writer, embedding_type, batch_size=1024, threshold=None, max_grad_norm=2.):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold
        
        if embedding_type == 'simple' or embedding_type == 'linear':
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
            self.val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_batch)
            self.val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_batch)

        self.actor_optim   = optim.Adam(model.actor.parameters(), lr=1e-4)
        self.max_grad_norm = max_grad_norm
        
        self.train_tour = []
        self.val_tour   = []
        
        self.epochs = 0
        self.best_model_path = best_model_path
        self.writer = writer
        self.embedding_type = embedding_type
        self.best_val_reward = sys.float_info.max
    
    def train_and_validate(self, n_epochs):

        beta = Config.BETA

        critic_exp_mvg_avg = torch.zeros(1)
        if Config.USE_CUDA: 
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

        for epoch in range(n_epochs):
            
            print(f'epoch {epoch} out of {n_epochs}')
            print('-------------------------------')
            print('train')
            for batch_id, sample_batch in enumerate(self.train_loader):
                self.model.train()
                if self.embedding_type == 'simple' or self.embedding_type == 'linear':
                  inputs = Variable(sample_batch)
                  inputs = inputs.cuda()
                  import copy
                  sample_batch_input = copy.deepcopy(inputs)
                  R, probs, actions, actions_idxs = self.model(inputs, sample_batch_input)

                else:
                  sample_batch_emb, sample_batch_input = torch.Tensor(sample_batch[0]), torch.Tensor(sample_batch[1])
                  inputs = Variable(sample_batch_emb)
                  inputs = inputs.cuda()
                  sample_batch_input = Variable(sample_batch_input)
                  sample_batch_input = sample_batch_input.cuda()

                  R, probs, actions, actions_idxs = self.model(sample_batch_input, inputs)

                if batch_id == 0:
                    critic_exp_mvg_avg = R.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

                advantage = R - critic_exp_mvg_avg

                logprobs = 0
                for prob in probs: 
                    logprob = torch.log(prob)
                    logprobs += logprob
                logprobs[logprobs < -1000] = 0.  

                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.actor.parameters(),
                                    float(self.max_grad_norm), norm_type=2)

                self.actor_optim.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                self.train_tour.append(R.mean().data.item())

                
            

                # visualisation input -----------
                if batch_id == 0:
                    
                    print(len(actions), actions[0].shape)
                    render(sample_batch_input, actions, R, [0, 1], 'train')
                # -------------------------------

                
                    
                        # -------------------------------
                # TENSORBOARD will be better for this plotting
                # if batch_id % 10 == 0:
                #     self.plot(self.epochs)
            print('-----------------------')
            print('evaluation')
            self.model.eval()
            for val_batch_id, val_batch in enumerate(self.val_loader):
                if self.embedding_type == 'simple' or self.embedding_type == 'linear':
                    inputs = Variable(sample_batch)
                    inputs = inputs.cuda()
                    import copy
                    val_batch_input = copy.deepcopy(inputs)
                    R, probs, actions, actions_idxs = self.model(inputs, val_batch_input)
                
                else:  
                    val_batch_emb, val_batch_input = torch.Tensor(val_batch[0]), torch.Tensor(val_batch[1])
                    inputs = Variable(val_batch_emb)
                    inputs = inputs.cuda()

                    val_batch_input = Variable(val_batch_input)
                    val_batch_input = val_batch_input.cuda()

                    R, probs, actions, actions_idxs = self.model(val_batch_input, inputs)
                self.val_tour.append(R.mean().data.item())

                # visualisation input -----------
                if (val_batch_id == 0) and (batch_id == 0):
                    render(val_batch_input, actions, R, [0, 1], 'validation')
            mean_val_reward = np.array(self.val_tour).mean()
            if self.writer is not None:
                  phase='train'
                  self.writer.add_scalar("train_reward/{}".format(phase), np.array(self.train_tour).mean(), epoch)
                  phase='val'
                  self.writer.add_scalar("test_reward/{}".format(phase), mean_val_reward, epoch)
            
            if mean_val_reward < self.best_val_reward:
                self.best_val_loss = mean_val_reward
                torch.save(self.model.state_dict(), self.best_model_path)


            if self.threshold and self.train_tour[-1] < self.threshold:
                print ("EARLY STOPPAGE!")
                break
                
            self.epochs += 1
                
