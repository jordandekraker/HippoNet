
import gymnasium as gym
import math
import random, datetime, os, copy, time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import mem

import AtariSpace

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using " + str(device))



# conv2d design testing
# ideally out convolutions should work with the input image size to only produce integers here. This will help a lot in the ConvTranspose2d decoder!
current_frame = torch.zeros((241,153)).to(device)
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1
    w = ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1
    return h, w
t1 = conv_output_shape(current_frame.shape,kernel_size=5, stride=2)
t2 = conv_output_shape(t1,kernel_size=3, stride=2)
t3 = conv_output_shape(t2,kernel_size=3, stride=2)
print(t1)
print(t2)
print(t3)
flatsz = int(np.prod(t3)*32)
print(flatsz)



class VNet(nn.Module):
    #This will be a convolutional AE with tied encoder-decoder weights. Encoder can optionally be frozen.

    def __init__(self, output_dim=(512)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2,bias=False),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2,bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2,bias=False),
            nn.Flatten(),
            nn.Linear(flatsz, 10000, bias=False),
            nn.Tanh(),
            nn.Linear(10000, output_dim, bias=False),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 10000, bias=False),
            nn.Tanh(),
            nn.Linear(10000,flatsz,bias=False),
            # nn.Unflatten(1,(1,241,153)),
            nn.Unflatten(1,(32,int(t3[0]),int(t3[1]))),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2,bias=False),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,bias=False),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=2,bias=False),
        )
            
        # tie the weights
        for i in range(len(self.encoder)):
            ii = len(self.encoder)-i-1
            if hasattr(self.encoder[i],'weight'):
                if len(self.encoder[i].weight.shape)>2:
                    self.decoder[ii].weight = nn.Parameter(self.encoder[i].weight)
                else:
                    self.decoder[ii].weight = nn.Parameter(self.encoder[i].weight.t())

        # frozen versions
        self.encoderfrozen = copy.deepcopy(self.encoder)
        for p in self.encoderfrozen.parameters():
            p.requires_grad = False
        self.decoderfrozen = copy.deepcopy(self.decoder)
        for p in self.decoderfrozen.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.encoder(input)
        elif model == "frozen":
            return self.encoderfrozen(input)
        
    def backward(self, input, model):
        if model == "online":
            return self.decoder(input)
        elif model == "frozen":
            return self.decoderfrozen(input)
vnet = VNet().to(device)
print(vnet)



VLoss = nn.MSELoss()
VOptim = optim.Adam(vnet.parameters())



hnet = mem.MemoryUnit(bank_dim=2000, fea_dim=521).to(device)
print(hnet)


HLoss = nn.MSELoss()
HOptim = optim.SGD(hnet.parameters(), lr=0.05)



class MNet(nn.Module):
    # simple fully connected with softmax at the end; optionally frozen
    
    def __init__(self, input_dim=(512), output_dim=(AtariSpace.possible_actions)):
        super().__init__()
        self.online = nn.Sequential(
            nn.Linear(input_dim, 512),
            #nn.Tanh(),
            nn.Linear(512, output_dim),
            nn.Softmax(dim=1),
        )

        self.frozen = copy.deepcopy(self.online)
        for p in self.frozen.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "frozen":
            return self.frozen(input)
        
mnet = MNet().to(device)
print(mnet)



m=0
def erlang(x,k=4,l=4):
    return 1-((l**k)*(x**(k-1))*(math.e**(-l*x)))/math.factorial(k-1)*2 -.4
def MLoss(action,x):
    global m
    mloss = torch.sum(action)*(x-m) # note action sums to 1
    m = (0.99*m)+(0.01*x) # "sticky" running average so we know if x was good relative to its temporal nieghbours
    return mloss
x = np.linspace(0,5,num=50)
plt.plot(x,erlang(x))
plt.title("Goldilocks novelty zone")
plt.xlabel("novelty (prediction error)")
plt.ylabel("motor policy loss")

MOptim = optim.SGD(mnet.parameters())



# initialize state
frame = AtariSpace.AtariPress(np.random.randint(0,AtariSpace.possible_actions))
current_LV = vnet.forward(AtariSpace.current_frame, model='frozen')

# the complicated part.
# 0) reconstruct the frame from the current LV (just for viewing)
# 1a) choose next action from the current LV
# 1b) make a prediction from the current LV about the next LV
# 1c) reconstruct the next frame from the predicted LV
# 2) take the action and get the actual next frame+LV
# 3a) train VNet to reconstruct the current frame from the current LV
# 3b) train HNet to predict next LV from the current LV
# 3c) train MNet to maximize HLoss
log = []
t = time.time()
for i in range(100000):
    
    current_action = mnet.forward(current_LV.detach(), model='online')
    x = torch.cat((current_LV.detach(), current_action.detach()),dim=1)
    x = x[None,:]
    predicted_LV = hnet(x)
    recon_frame = vnet.backward(current_LV.detach(), model='online')
    predicted_frame = vnet.backward(predicted_LV[:,0,:512].detach(), model='online')
    
    vloss = VLoss(recon_frame,AtariSpace.current_frame)
    VOptim.zero_grad()
    vloss.backward()
    VOptim.step()
    
    old_frame = copy.deepcopy(AtariSpace.current_frame.detach())
    if np.random.rand()<0.05:
        a = np.random.randint(0,AtariSpace.possible_actions)
    else:
        a = np.argmax(current_action.to('cpu').detach())
    frame = AtariSpace.AtariPress(a)
    next_LV = vnet.forward(AtariSpace.current_frame.detach(), model='online')
    if i%1==0:
        AtariSpace.live_plot([old_frame,recon_frame,AtariSpace.current_frame,predicted_frame])
        print(i)
   
    y = torch.cat((next_LV.detach(), current_action.detach()),dim=1)
    y = y[None,:]
    hloss = HLoss(predicted_LV,y)
    HOptim.zero_grad()
    hloss.backward()
    HOptim.step()
    mloss = -MLoss(current_action,hloss.detach()/512)
    MOptim.zero_grad()
    mloss.backward()
    MOptim.step()
    
    log.append([vloss.to('cpu').detach(), hloss.to('cpu').detach(), mloss.to('cpu').detach(), a, AtariSpace.g])
    current_LV = next_LV
    del predicted_LV, recon_frame, predicted_frame, next_LV, hloss, vloss, mloss, x, y

elapsed = time.time() - t
print(elapsed)



plt.figure(figsize=(16, 5), dpi=80)
smoothwindow=100
log = np.asarray(log)
plt.subplot(1,5,1)
plt.plot(np.convolve(np.array(log)[:,0],np.ones(smoothwindow)/smoothwindow, mode='valid'))
plt.subplot(1,5,2)
plt.plot(np.convolve(np.array(log)[:,1],np.ones(smoothwindow)/smoothwindow, mode='valid'))
plt.subplot(1,5,3)
plt.plot(np.convolve(np.array(log)[:,2],np.ones(smoothwindow)/smoothwindow, mode='valid'))
plt.subplot(1,5,4)
plt.scatter(range(len(log)),log[:,3])
plt.subplot(1,5,5)
plt.scatter(range(len(log)),log[:,4])