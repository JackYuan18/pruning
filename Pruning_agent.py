
import torch
from models import *
import matplotlib.pyplot as plt
from synthetic import latent_function
import torch.nn as nn
import tqdm

import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# import matplotlib
# matplotlib.use("Agg")

class Pruning:
    
    def __init__(self, originial_net_path, device,
                 neuron_pruning_threshold = 0.001,
                 model_loss_threshold = 0.05,
                 layer_pruning_threshold = 0.5,
                 METRIC_LOG_EVERY = 10,
                 lr = 0.003):

        self.dnn_path = originial_net_path
        
        self.dnn = torch.load(originial_net_path, weights_only = False)
        torch.save(self.dnn, 'dnn_before_prune.pt')
        self.dnn.train()
        self.hookup(self.dnn)
        
        self.neuron_pruning_threshold = neuron_pruning_threshold
        self.model_loss_threshold = model_loss_threshold
        self.layer_pruning_threshold = layer_pruning_threshold
        self.METRIC_LOG_EVERY = METRIC_LOG_EVERY
        
        self.dnn.to(device)
        self.device = device
        
        self.mse = nn.MSELoss()
        self.optim = torch.optim.Adam(self.dnn.parameters(), lr = lr)

        
        self.modelLosses = []
        self.actLosses = []
        self.act_nums = []
        
        

    def unhook(self, dnn):
        for module in dnn.net.modules():
            if isinstance(module, nn.ReLU):
                module._forward_hooks.clear()
                
    def hookup(self, dnn):
        # remove_layer reuses existing ReLU modules; re-registering hooks without
        # clearing leaves stale hooks that still write old name keys (e.g. net.13).
        
        for name, layer in dnn.named_modules():
            if isinstance(layer, nn.ReLU):  # or Linear, Conv2d, etc.
                layer.register_forward_hook(self.get_activation(dnn, name))
                
    def get_activation(self, dnn, name):
        def hook(model, input, output):
            # Keep fraction as a tensor (no .item()) to avoid a GPU sync every forward.
            with torch.no_grad():
                frac = (output > self.neuron_pruning_threshold).float().mean()
            dnn.activations[name] = (output, frac)
        return hook
    
    def get_task_specific_data(self, X_task):
        with torch.no_grad():
            self.Y_origin = self.dnn(X_task)
        self.X_task = X_task
        self.dnn(X_task)
        self.dnn.X = X_task
    
    
    def neuron_sparification(self, iter_num = 3000):
        for inner in range(iter_num):
            model_loss = self.mse(self.dnn(self.X_task), self.Y_origin)
            act_loss = sum(a[0].abs().mean() for a in self.dnn.activations.values())
            
            if inner % self.METRIC_LOG_EVERY == 0:
                self.modelLosses.append(model_loss.detach().item())
                self.actLosses.append(act_loss.detach().item())
                self.act_nums.append(sum(p.numel() for p in self.dnn.parameters()))
            
            # loss = act_loss + 0.1*torch.log(0.05-model_loss)
            
            self.optim.zero_grad()    
            if model_loss > self.model_loss_threshold:
                model_loss.backward()
            else:
                act_loss.backward()
            
            self.optim.step()
        return model_loss, act_loss
            
    def layer_pruning(self, model_loss):
        pruned = False
        if model_loss <= self.model_loss_threshold:
            act_vals = list(self.dnn.activations.values())
            ratios = torch.stack([a[1] for a in act_vals])
            val, remove_layer_idx = torch.min(ratios, 0)
            remove_layer_idx = int(remove_layer_idx)
            
            #Either prune layer or prune neurons
            if act_vals[remove_layer_idx][1].item() < self.layer_pruning_threshold and 0 < remove_layer_idx < len(self.dnn.net) - 1:
                
                self.unhook(self.dnn)
                torch.save(self.dnn, 'dnn_before_prune.pt')
                self.dnn.remove_layer(remove_layer_idx)
                self.hookup(self.dnn)
                print(f'Layer pruned...{remove_layer_idx}')
                pruned = True
        return pruned
        
        
    def neuron_pruning(self, model_loss, layer_pruned):
        if model_loss <= self.model_loss_threshold and not layer_pruned:
            _ = self.dnn(self.X_task)
            
            linear_idx = [i for i, m in enumerate(self.dnn.net) if isinstance(m, nn.Linear)]
            act_list = [
                self.dnn.activations[k]
                for k in sorted(self.dnn.activations.keys(), key=lambda n: int(n.split('.')[-1]))
            ]
            n_pairs = min(len(linear_idx) - 1, len(act_list))
            # print(f'Neuron pruning...{len(dnn.activations)} (pairs {n_pairs})')
            for j in range(n_pairs):
                a = act_list[j]
                keep_idx = torch.where(a[0].abs().mean(dim=0) > self.neuron_pruning_threshold)[0]
                if len(keep_idx) == 0:
                    keep_idx = torch.tensor([0], device=a[0].device, dtype=torch.long)
                li, lj = linear_idx[j], linear_idx[j + 1]
                layer = self.dnn.net[li]
                next_layer = self.dnn.net[lj]
                dev = layer.weight.device
                dtype = layer.weight.dtype
                in_dim = layer.in_features
                new_layer = nn.Linear(in_dim, len(keep_idx)).to(device=dev, dtype=dtype)
                new_next_layer = nn.Linear(len(keep_idx), next_layer.out_features).to(
                    device=dev, dtype=dtype
                )
                with torch.no_grad():
                    new_layer.weight.copy_(layer.weight[keep_idx, :])
                    new_layer.bias.copy_(layer.bias[keep_idx])
                    new_next_layer.weight.copy_(next_layer.weight[:, keep_idx])
                    new_next_layer.bias.copy_(next_layer.bias)
                self.dnn.net[li] = new_layer
                self.dnn.net[lj] = new_next_layer
        
    def evaluation(self, X_all,X_task_test):
        self.dnn.eval()
        
        Y_pred_all = self.dnn(X_all)
        dnn_origin = torch.load(self.dnn_path, weights_only = False)
        dnn_origin.to(self.device)
        dnn_origin.eval()
        Y_pred_all_static = dnn_origin(X_all)

        plt.figure()
        plt.plot(self.modelLosses)
        plt.plot(self.act_nums, color = 'red')

        plt.figure()

        plt.plot(X_all.detach().numpy(), Y_pred_all_static.detach().numpy(),color='blue')
        plt.plot(X_all.detach().numpy(), Y_pred_all.detach().numpy(),'--',color='red')
        
        plt.show()
        
        
        Y_pred = self.dnn(X_task_test).detach().numpy()
        Y_pred_origin = dnn_origin(X_task_test).detach().numpy()
        Y_truth = latent_function(X_task_test.numpy())
        
        pred_error = np.mean((Y_pred - Y_truth)**2)
        pred_error_origin = np.mean((Y_pred_origin - Y_truth)**2)
        
        print(f'Task-specific testing error after pruning: {pred_error}')
        print(f'Task-specific testing error before pruning: {pred_error_origin}')       
        
        print(f'Neuron prunning threshold: {self.neuron_pruning_threshold}')
        print(f'Layer pruning threshold: {self.layer_pruning_threshold}')
        print(f'Pruning_ratio: {self.act_nums[-1]/self.act_nums[0]}')
        print(f'Model loss: {self.modelLosses[-1]}')
        
    
    def prune_model(self, X_task, epoch_num=100):
        
        self.get_task_specific_data(X_task)
        
        model_loss = 0
        for epoch in tqdm.tqdm(range(100)):
            layer_pruned = self.layer_pruning(model_loss)
            self.neuron_pruning(model_loss, layer_pruned)
            
            self.neuron_sparification()


if __name__=='__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        # Fixed input shape (1000x1) — lets cuDNN pick fast algorithms.
        torch.backends.cudnn.benchmark = True
        
    X_all = torch.linspace(0,10,5000, device=device)
    X_all = X_all.unsqueeze(1)

    X_task = torch.empty_like(X_all)
    X_task.uniform_(3, 7)
    train_num = int(0.9*len(X_task))
    
    X_task_train = X_task[:train_num]
    X_task_test = X_task[train_num:]
    
    PATH = 'dnn_synthetic_data.pt'
    pruning_agent = Pruning(PATH, device)
    
    pruning_agent.prune_model(X_task_train)
    
    pruning_agent.evaluation(X_all, X_task_test)
    
