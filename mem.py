# https://github.com/donggong1/memae-anomaly-detection/blob/master/models/memae_3dconv.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class MemoryUnit(nn.Module):
    def __init__(self, fea_dim, bank_dim=20, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.bank_dim = bank_dim
        self.fea_dim = fea_dim
        self.bank = nn.Parameter(torch.Tensor(self.bank_dim, self.fea_dim), requires_grad=True)  # Note: gradient needs to be computed if this will be used to optimize VNet (low lr). We will ALSO update here with a high lr
        self.banklr = 0.5
        self.threshold_memUpdate = 0.5 # if similarity is greater than this threshold, update the memory slot. else, LRUA
        self.bias = None
        self.shrink_thres= shrink_thres
        self.recency = torch.Tensor(self.bank_dim)
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.bank.size(1))
        self.bank.data.uniform_(-stdv, stdv)
        self.recency.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        mem_trans = self.bank.t()  # Mem^T, MxC
        att_weight = F.linear(input, self.bank)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            # att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            # att_weight = F.normalize(att_weight, p=1, dim=1)
            att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        output = torch.tanh(F.linear(att_weight, mem_trans))  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        
        ### bank update rule ###
        
        ### all head update
        outback = output[0,:]
        # self.bank.data = self.bank*(1-self.banklr) + (outback * att_weight.reshape(1,-1).t() * self.banklr)
        
        ### single head update
        memind = torch.argmax(outback).item()
        #self.bank.data[memind,:] = self.bank[memind,:]*(1-self.banklr) + (outback * self.banklr) # update only the most accessed slot

        ### LRUA update (https://arxiv.org/pdf/1605.06065?)
        LRUA = torch.argmin(self.recency)
        ind = memind if torch.max(output[0,:]) > self.threshold_memUpdate else LRUA
        self.recency[memind] +=1
        self.recency -= 1/self.bank_dim
        self.bank.data[ind,:] = self.bank[ind,:]*(1-self.banklr) + (outback * self.banklr) # update only the least recently used access.
        
        ## successive slots update rule (NTM) for continuous timeseries
        
        return output #{'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'bank_dim={}, fea_dim={}'.format(
            self.bank_dim, self.fea_dim
        )

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output