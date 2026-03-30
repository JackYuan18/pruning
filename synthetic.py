# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:34:02 2026

@author: yzyja
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from typing import List
import tqdm
from models import *
torch.manual_seed(0)
np.random.seed(0)
def latent_function(x):
    return np.piecewise(
        x,
        [(0 <= x) & (x < 3),
         (3 <= x) & (x < 7),
         x >= 7],
        [lambda x: 0.5*x,
         lambda x: -2*x+5,
         lambda x: 1.5*x-10]
    )








def dataloader(X,Y,batch_size):
    for start in range(0,len(X), batch_size):
        X_batch = X[start:start+batch_size]
        Y_batch = Y[start:start+batch_size]
        yield X_batch, Y_batch
activations = {}    
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook
if __name__ == '__main__':
    
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    data_num = 100000
    batch_size = 128
    
    X = np.random.uniform(0,10,(data_num,1))
    eps = np.random.normal(0,0.01,(data_num,1))
    Y = latent_function(X) + eps


    train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2

    X_train = X[:int(data_num*train_ratio)]
    Y_train = Y[:int(data_num*train_ratio)]

    X_val = X[int(data_num*train_ratio): int(data_num*train_ratio)+int(data_num*val_ratio)]
    Y_val = Y[int(data_num*train_ratio): int(data_num*train_ratio)+int(data_num*val_ratio)]

    X_test = X[-int(data_num*test_ratio):]
    Y_test = Y[-int(data_num*test_ratio):]
    


    X_train  = torch.Tensor(X_train, device = device)

    Y_train  = torch.Tensor(Y_train, device = device)

    X_val = torch.Tensor(X_val, device = device)
    Y_val = torch.Tensor(Y_val, device = device)

    X_test = torch.Tensor(X_test, device = device)
    Y_test = torch.Tensor(Y_test, device = device)

    train_data = dataloader(X_train, Y_train, batch_size)
    
    dnn = DNN(1,1, [8,16,64,128,64,16,8]).to(device)
    hooks = []
    for name, layer in dnn.named_modules():
        if isinstance(layer, torch.nn.ReLU):  # or Linear, Conv2d, etc.
            hooks.append(layer.register_forward_hook(get_activation(name)))


    optim = torch.optim.Adam(dnn.parameters(), lr = 0.001)
    loss_fn = torch.nn.MSELoss()
    
    epoch_num = 2000
    losses = []
    data_losses = []
    act_losses = []
    val_losses = []
    
    for epoch in tqdm.tqdm(range(epoch_num)):
        dnn.train()
        for X_batch, Y_batch in train_data:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            Y_pred = dnn(X_batch)
            
            optim.zero_grad()
            act_loss = sum(a.abs().mean() for a in activations.values())
            data_loss = loss_fn(Y_pred, Y_batch)
            loss = data_loss #+ 0.01*act_loss
            losses.append(loss.item())
            data_losses.append(data_loss.item())
            act_losses.append(act_loss.item())
            loss.backward()
            optim.step()
        dnn.eval()
        with torch.no_grad():
            Y_val_pred = dnn(X_val)
            val_loss = loss_fn(Y_val_pred,Y_val)
            val_losses.append(val_loss.item())
    
    plt.figure()
    plt.plot(data_losses)
    plt.plot(act_losses, color = 'red')
    dnn.eval()
    with torch.no_grad():
        Y_test_pred = dnn(X_test)        
        test_loss = loss_fn(Y_test,Y_test_pred)
    print(test_loss)
    
    X_all = torch.linspace(0,10,1000)
    X_all = X_all.unsqueeze(1)
    Y_pred_all = dnn(X_all)
    
    plt.figure()
    plt.scatter(X,Y)
    
    plt.plot(X_all.detach().numpy(), Y_pred_all.detach().numpy(),color='red')
    
    for h in hooks:
        h.remove()
    PATH = 'dnn_synthetic_data.pt'
    torch.save(dnn, PATH)
        
        
    
    
    
    
    
            
