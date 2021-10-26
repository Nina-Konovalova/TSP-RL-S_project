import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

import sys, os
sys.path.append("../")


import warnings
warnings.simplefilter("ignore")

from model.attention import Attention
import Config
from embeddings.GraphEmbedding import GraphEmbedding
from embeddings.LinearGraphEmbedding import LinearGraphEmbedding

class PointerNet(nn.Module):
    def __init__(self, 
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            attention,
            embeddings_type,
            use_cuda=Config.USE_CUDA):

        super(PointerNet, self).__init__()
        
        self.embedding_size  = embedding_size
        self.hidden_size     = hidden_size
        self.n_glimpses      = n_glimpses
        self.seq_len         = seq_len
        self.use_cuda        = use_cuda
        self.embeddings_type = embeddings_type
        
        if self.embeddings_type == 'simple':
            self.embedding = GraphEmbedding(2, embedding_size, use_cuda=use_cuda)
        elif self.embeddings_type == 'linear':
            self.embedding = LinearGraphEmbedding(2, embedding_size, use_cuda=use_cuda)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention, use_cuda=use_cuda)
        self.glimpse = Attention(hidden_size, use_tanh=False, name=attention, use_cuda=use_cuda)
        
        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        
    def apply_mask_to_logits(self, logits, mask, idxs): 
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask
            
    def forward(self, inputs):
        """
        Args: 
            inputs: [batch_size x 1 x sourceL]
        """
        batch_size = inputs.size(0)
        
        if self.embeddings_type == 'simple' or self.embeddings_type == 'linear':
            seq_len    = inputs.size(2)
            assert seq_len == self.seq_len
            embedded = self.embedding(inputs)

        else:
            seq_emb    = inputs.size(2)
            assert seq_emb == self.embedding_size

            seq_len    = inputs.size(1)
            assert seq_len == self.seq_len

            import copy
            embedded = copy.deepcopy(inputs)
        

        encoder_outputs, (hidden, context) = self.encoder(embedded)
        
        
        prev_probs = []
        prev_idxs = []
        mask = torch.zeros(batch_size, seq_len).byte()
        if self.use_cuda:
            mask = mask.cuda()
            
        idxs = None
       
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        
        for i in range(seq_len):
            
            
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
            
            query = hidden.squeeze(0)
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2) 
                
                
            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits)
            
            
            idxs = probs.multinomial(1).squeeze(1)
            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    print(seq_len)
                    print(' RESAMPLE!')
                    idxs = probs.multinomial(1).squeeze(1)
                    break
            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :] 
            
            prev_probs.append(probs)
            prev_idxs.append(idxs)
            
        return prev_probs, prev_idxs