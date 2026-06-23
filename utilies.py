## tools for HT-NN training & CRF generation
import os
import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from torch.utils.data import Dataset

## Pytorch training classes
class NNDataset(Dataset):
   
   def __init__(self, work_dir, name):

      with h5py.File(os.path.join(work_dir, str('head_' + name  + '_X.hdf5')), 'r') as f:
                  x = f['dataset'][()].astype(np.float32)
      with h5py.File(os.path.join(work_dir, str('material_' + name + '_Y.hdf5')), 'r') as f:
                  y = f['dataset'][()].astype(np.float32)     

      self.x = torch.FloatTensor(x)
      self.y = torch.FloatTensor(y)

   def __len__(self):    
         return self.x.shape[0]

   def __getitem__(self, index):
         return self.x[index], self.y[index] 
     
class _Dense_layer(nn.Sequential):
    def __init__(self,
                 in_features, 
                 growth_rate,
                 kernel_size, stride, padding,   
                 drop_rate = 0., 
                 ):
        super(_Dense_layer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('leakyrelu1', nn.LeakyReLU(negative_slope=0.3))  
        self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                                           kernel_size = kernel_size,
                                           stride = stride,
                                           padding = padding,
                                           bias = False))
        self.drop_rate = drop_rate
    def forward(self, x):

        y = super(_Dense_layer, self).forward(x)
        if self.drop_rate > 0:
            y = F.dropout2d(y, p = self.drop_rate, training = self.training)
        z = torch.cat([x,y], 1)
        return z
    
class _Dense_block(nn.Sequential):
    def __init__(self,
                 num_layers, # 經過幾個denselayer(concatenate 幾次)
                 in_features, # 輸入的channel數量
                 growth_rate, # concatenate 的層數
                 kernel_size, stride, padding,# 定義denselayer內部的conv參數(預設(3,1,1))
                 drop_rate = 0.,   
                 ):
        super(_Dense_block, self).__init__()

        # self.features = nn.Sequential() # 測試用

        for i in range(num_layers):
            layer = _Dense_layer(in_features + i*growth_rate, # 更改每一次的輸入的features 數量(每經過一個dl, 加一組growth_rate)
                                 growth_rate,
                                 kernel_size, stride, padding, drop_rate)

            self.add_module('denselayer%d'%(i+1),layer)
            
class _Transition_encoder(nn.Sequential):
        def __init__(self,
                     in_features,
                     out_features,
                     conv_args, # tuple (kernel_size, stride, padding)  
                     drop_rate = 0.
                     ):
            super(_Transition_encoder, self).__init__()

            # Reduce input features(from in_features => out_features)
            self.add_module('norm1', nn.BatchNorm2d(in_features)) 
            self.add_module('leakyrelu1', nn.LeakyReLU(negative_slope=0.3))
            self.add_module('conv_shrink_features', nn.Conv2d(in_features, out_features,
                                                          kernel_size = 1, stride = 1, padding = 0, 
                                                          bias = False))
            # Adjust(Reduce) image size
            if drop_rate > 0: 
                self.add_module('dropout1', nn.Dropout2d(p = drop_rate))
            self.add_module('norm2', nn.BatchNorm2d(out_features))
            self.add_module('leakyrelu2', nn.LeakyReLU(negative_slope=0.3))
            self.add_module('conv_encoding',nn.Conv2d(out_features, out_features,
                                                          kernel_size = conv_args[0], stride = conv_args[1], padding = conv_args[2], 
                                                          bias = False) )
            if drop_rate > 0: 
                self.add_module('dropout2', nn.Dropout2d(p = drop_rate))

class _Transition_decoder(nn.Sequential):
        def __init__(self,
                     in_features,
                     out_features,
                     convt_arg,    
                     drop_rate = 0.,
                     ):
            super(_Transition_decoder, self).__init__()
            
            # Reduce input features & adjust size by maxpooling
            self.add_module('norm1', nn.BatchNorm2d(in_features)) 
            self.add_module('leakyrelu1', nn.LeakyReLU(negative_slope=0.3))
            self.add_module('conv_shrink_features', nn.Conv2d(in_features, out_features,
                                                          kernel_size = 1, stride = 1, padding = 0, 
                                                          bias = False))
            
            if drop_rate > 0: 
                self.add_module('dropout1', nn.Dropout2d(p = drop_rate))

            self.add_module('norm2', nn.BatchNorm2d(out_features))
            self.add_module('leakyrelu2', nn.LeakyReLU(negative_slope=0.3))
            self.add_module('convt_decoding', nn.ConvTranspose2d(out_features, out_features, kernel_size = convt_arg[0], stride = convt_arg[1], padding = convt_arg[2], bias = False))

        
            if drop_rate > 0: 
                self.add_module('dropout2', nn.Dropout2d(p = drop_rate))
                
class Dense_Transition_Encoder_block(nn.Module):
    def __init__(self, in_channels, 
                 growth_rate, 
                 num_layers, 
                 conv_args, 
                 drop_rate=0.):
        super().__init__()
        self.dense_block = _Dense_block(num_layers, in_channels, growth_rate,
                                        kernel_size=3, stride=1, padding=1, drop_rate=drop_rate)
        dense_out = in_channels + growth_rate * num_layers
        self.transition = _Transition_encoder(dense_out, dense_out//2,
                                              conv_args=conv_args, drop_rate=drop_rate)
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.transition(x)
        return x
    
class Dense_Transition_Decoder_block(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, convt_arg, drop_rate=0.):
        super().__init__()
        self.dense_block = _Dense_block(num_layers, in_channels, growth_rate,
                                        kernel_size=3, stride=1, padding=1, drop_rate=drop_rate)
        dense_out = in_channels + growth_rate * num_layers
        self.transition = _Transition_decoder(dense_out, dense_out//2, convt_arg = convt_arg, drop_rate=drop_rate)
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.transition(x)

        return x
    
class Activation_block(nn.Module):
    def __init__(self, negative_slope):
        super(Activation_block, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=False)


##  CRF generation
def parse_estimation(file_name, dim):
    """
    Used for parsing SLE inversion result, prepared for random field generation step
    Args:
        file_name: file name(direct path) for O-kestimate.dat or O-sestimate.dat 
        dim: 2d or 3d
    Returns:
        df_result_ln_mean: SLE estimated mean (in log)
        df_result_ln_var: SLE estimated variance (in log)
    """
    
    file_name = Path(file_name)  
    base_dir = file_name.parent
    
    with open(file_name, 'r') as file:
        lines = file.readlines()

    df = pd.read_csv( base_dir /'grid.dat', header=None,sep='\s+', engine='python')    
    ele_num = df.iloc[0,0]
    node_num = df.iloc[1,0]

    idx_list = []
    col_name = []
    iteration = 1
    for idx, line in enumerate(lines):
        if 'zone' in line:
            col_name.append('iter_' + str(iteration))
            idx_list.append(idx)
            iteration += 1    

    column_data = []  
    column_ln_mean = []
    column_ln_var = []

    for i in range(len(idx_list)):
        str_idx = idx_list[i] + 1

        # estimation mean
        if i == 0:
            var_total_length = (node_num*dim + ele_num*2)
            mean_total_length = (node_num*dim + ele_num*3) 
        else:
            var_total_length = ele_num*2
            mean_total_length = ele_num*3

        mean_section = lines[str_idx: str_idx + mean_total_length]
        var_section = lines[str_idx: str_idx + var_total_length]

        mean_data = mean_section[-ele_num:]
        ln_var_data = var_section[-ele_num:]

        mean_data = [float(item.strip()) for item in mean_data]
        ln_var_data = [float(item.strip()) for item in ln_var_data]
        column_ln_mean.append(mean_data)
        column_ln_var.append(ln_var_data)

    column_ln_mean = np.array(np.log(column_ln_mean)).T 
    column_ln_var = np.array(column_ln_var).T 
    column_data = np.array(column_data).T 
    df_result_mean = pd.DataFrame(columns=col_name, data =column_ln_mean)
    df_result_var = pd.DataFrame(columns=col_name, data =column_ln_var)

    df_gird = pd.read_csv( base_dir /'grid-material.dat', sep='\s+', names = ['idx', 'x', 'y', 'z']) # ['idx', 'x', 'y', 'z]
    df_gird = df_gird.drop(columns='idx')
    df_result_ln_mean = pd.concat([df_gird,df_result_mean],axis=1)
    df_result_ln_var = pd.concat([df_gird,df_result_var],axis=1)


    return df_result_ln_mean, df_result_ln_var

