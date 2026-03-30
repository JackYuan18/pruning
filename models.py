# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 23:41:49 2026

@author: yzyja
"""
import torch.nn as nn
from typing import List
import torch
import torch.nn.functional as F
class DNN(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dims: List[int]=[8,], neuron_pruning_threshold=0.001):
        super().__init__()
        hidden_layers = []
        hidden_layers.append(nn.Linear(x_dim,hidden_dims[0]))
        hidden_layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_dims)):
            hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            hidden_layers.append(nn.ReLU())
        
        hidden_layers.append(nn.Linear(hidden_dims[-1], y_dim))
        self.net = nn.Sequential(*hidden_layers)
        self.activations = {}
        self.neuron_pruning_threshold = neuron_pruning_threshold
        
        self.X = None
    
    def unhook(self):
        for module in self.net.modules():
            if isinstance(module, nn.ReLU):
                module._forward_hooks.clear()
    def hookup(self):
        # remove_layer reuses existing ReLU modules; re-registering hooks without
        # clearing leaves stale hooks that still write old name keys (e.g. net.13).
        
        for name, layer in self.named_modules():
            if isinstance(layer, nn.ReLU):  # or Linear, Conv2d, etc.
                layer.register_forward_hook(self.get_activation(name))


    def forward(self,x, return_activations = False):
        return self.net(x)
    
    def get_activation(self, name):
        def hook(model, input, output):
            # Keep fraction as a tensor (no .item()) to avoid a GPU sync every forward.
            with torch.no_grad():
                frac = (output > self.neuron_pruning_threshold).float().mean()
            self.activations[name] = (output, frac)
        return hook
    def remove_layer(self, idx, train_steps = 500, lr = 0.001):
        layer = self.net[2*idx]
        layer_front = self.net[2*(idx-1)]
        
        dev = layer_front.weight.device
        dtype = layer_front.weight.dtype

        new_layer_front = nn.Linear(in_features = layer_front.in_features, 
                                    out_features = layer.out_features).to(device=dev, dtype=dtype)
        
        
        # with torch.no_grad():
        print(idx)
        if idx == 1:
            layer_front_activations = self.X
        else:
            layer_front_activations = self.activations[f'net.{2*(idx-1)-1}'][0].detach()
        layer_back_activations = self.activations[f'net.{2*(idx+1)-1}'][0].detach()
        opt = torch.optim.Adam(new_layer_front.parameters(), lr=lr)
        for _ in range(train_steps):
           
            pred = F.relu(new_layer_front(layer_front_activations))
            loss = F.mse_loss(pred, layer_back_activations)
            opt.zero_grad()
            loss.backward()
            opt.step()


        layers = list(self.net.children())
        
        new_layers = (
                    layers[:2*(idx-1)] + 
                    [new_layer_front, nn.ReLU()] +
                    layers[2*(idx+1):]
                    )
        self.net = nn.Sequential(*new_layers)
        self.activations = {}
        # self.unhook()
        # self.hookup()

