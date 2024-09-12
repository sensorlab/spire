from torch import nn
import torch
import numpy as np
from typing import Tuple

class Simple_CNN_1D(nn.Module):
    
    name = "Simple_CNN_1D"
    
    def __init__(
        self, 
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_signal_length:int=1024, 
        in_channels:int=1,  
        kernel_size:int=7, 
        n_classes:int=10, 
        padding_size:int=3,
        features_size:int=100):
        
        super(Simple_CNN_1D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.input_signal_length = input_signal_length
        self.layers_output_sizes = layers_output_sizes
        self.n_classes = n_classes
        self.features_size = features_size
        
        self.feature_layers = self._build_feature_layers(
            layers_output_sizes=layers_output_sizes, 
            input_signal_length=input_signal_length, 
            in_channels=in_channels,  
            kernel_size=kernel_size, 
            n_classes=n_classes, 
            padding_size=padding_size,
            features_size=features_size
        )
        
        self.classif_layers = nn.Sequential(
            nn.Linear(features_size, n_classes)
        )
        
    def _compute_conv1d_output_length(
        self,
        input_length:int,
        padding_size:int=3,
        kernel_size:int=7,
        stride:int=1
    ):
        """
        Compute the output's signal length of the shape (output_channels, output_length)
        """
        output_length = np.ceil(
            (input_length + 2 * padding_size - kernel_size) / stride  + 1
        )
        return int(output_length)
    
    def _compute_maxpool1d_output_length(
        self,
        input_length:int,
        padding_size:int=0,
        kernel_size:int=4,
        stride:int=4
    ):
        """
        Compute the output's signal length of the shape (output_channels, output_length)
        """
        output_length = np.ceil(
            (input_length + 2 * padding_size - kernel_size) / stride  + 1
        )
        return int(output_length)
    
        
    def _build_feature_layers(
        self,
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_signal_length:int=1024, 
        in_channels:int=1,  
        kernel_size:int=7, 
        n_classes:int=10, 
        padding_size:int=3,
        features_size:int=100)->nn.Sequential:
        """
        Build the feature map
        """
        # calc the result shape
        cur_signal_length = input_signal_length
        
        num_layers = len(layers_output_sizes)
        layers = []
        
        for i in range(num_layers):
            
            # input channels for current layer
            if i == 0:
                cur_in_channels = in_channels
            else:
                cur_in_channels = layers_output_sizes[i-1]
                
            # conv1d->batchnorm->relu->maxpool
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=cur_in_channels,
                        out_channels= layers_output_sizes[i], 
                        kernel_size=kernel_size, 
                        padding=padding_size),
                    nn.BatchNorm1d( layers_output_sizes[i]),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=4)
                )
            )
            
            # update the singal length
            cur_signal_length = self._compute_conv1d_output_length(
                input_length=cur_signal_length
            )
            cur_signal_length = self._compute_maxpool1d_output_length(
                input_length=cur_signal_length
            )
            
        # total size of the output tensor(num_chanhnels, signal_length)
        output_size = cur_signal_length * layers_output_sizes[-1] # default 512
        
        # linear layer for feature map reduction
        layers.extend(
            [
            nn.Flatten(),
            nn.Linear(output_size, features_size),
            ]
        )
        return nn.Sequential(*layers)
        
    def forward(self, x):
        features = self.feature_layers(x)
        return self.classif_layers(features)
    
    
    
class AE_CNN_1D(nn.Module):
    
    name = "AE_CNN_1D"
    
    def __init__(
        self, 
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_signal_length:int=1024, 
        in_channels:int=1,  
        kernel_size_conv:int=7, 
        kernel_size_maxpool:int=4,
        n_classes:int=10, 
        padding_size:int=3,
        features_size:int=10):
        
        super(AE_CNN_1D, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size_conv = kernel_size_conv
        self.kernel_size_maxpool = kernel_size_maxpool
        self.padding_size = padding_size
        self.input_signal_length = input_signal_length
        self.layers_output_sizes = layers_output_sizes
        self.n_classes = n_classes
        self.features_size = features_size

        # build feature map layers and memorize maxpoolings positions
        (self.feature_layers, self.maxpooling_layers_positions, 
        self.internal_signal_length) = self._build_feature_layers(
            layers_output_sizes=layers_output_sizes, 
            input_signal_length=input_signal_length, 
            in_channels=in_channels,
            kernel_size_conv=kernel_size_conv,
            kernel_size_maxpool=kernel_size_maxpool,
            n_classes=n_classes, 
            padding_size=padding_size,
            features_size=features_size
        )

        # build reconstruction layers: inverse order of feature map layers
        # memorize maxunpoolings positions
        self.reconstruction_layers, self.maxunpooling_layers_positions = \
        self._build_reconstruction_layers(
            layers_output_sizes=layers_output_sizes, 
            output_signal_length=self.internal_signal_length, 
            in_channels=in_channels,  
            kernel_size_conv=kernel_size_conv,
            kernel_size_maxpool=kernel_size_maxpool,
            n_classes=n_classes, 
            padding_size=padding_size,
            features_size=features_size
        )
        
    def _compute_conv1d_output_length(
        self,
        input_length:int,
        padding_size:int=3,
        kernel_size:int=7,
        stride:int=1
    )->int:
        """
        Compute the output's signal length of the shape (output_channels, output_length)
        """
        output_length = np.ceil(
            (input_length + 2 * padding_size - kernel_size) / stride  + 1
        )
        return int(output_length)
    
    def _compute_maxpool1d_output_length(
        self,
        input_length:int,
        padding_size:int=0,
        kernel_size:int=4,
        stride:int=4
    )->int:
        """
        Compute the output's signal length of the shape (output_channels, output_length)
        """
        output_length = np.ceil(
            (input_length + 2 * padding_size - kernel_size) / stride  + 1
        )
        return int(output_length)
    
        
    def _build_feature_layers(
        self,
        layers_output_sizes:list=[16, 32, 64, 128], 
        input_signal_length:int=1024, 
        in_channels:int=1,  
        kernel_size_conv:int=7, 
        kernel_size_maxpool:int=4, 
        n_classes:int=10, 
        padding_size:int=3,
        features_size:int=10)->Tuple[nn.ModuleList, set, int]:
        """
        Build the feature map, return maxpooling layers indices and internal signal length
        """
        # calc the result shape
        cur_signal_length = input_signal_length
        
        num_layers = len(layers_output_sizes)
        
        layers = []
        maxpooling_layers_positions = set()
        
        counter = 0
        
        for i in range(num_layers):
            
            # input channels for current layer
            if i == 0:
                cur_in_channels = in_channels
            else:
                cur_in_channels = layers_output_sizes[i-1]

            # memorize current maxpooling's position
            maxpooling_layers_positions.add(counter + 3)
            
            counter += 4

            # conv1d->batchnorm->relu->maxpool
            layers.extend(
                    [
                    nn.Conv1d(
                        in_channels=cur_in_channels,
                        out_channels= layers_output_sizes[i], 
                        kernel_size=kernel_size_conv, 
                        padding=padding_size),
                    nn.BatchNorm1d(layers_output_sizes[i]),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=kernel_size_maxpool,
                                 return_indices=True)
                    ]
            )

            # update the singal length
            cur_signal_length = self._compute_conv1d_output_length(
                input_length=cur_signal_length
            )
            cur_signal_length = self._compute_maxpool1d_output_length(
                input_length=cur_signal_length
            )
            
        # total size of the output tensor(num_chanhnels, signal_length)
        output_size = cur_signal_length * layers_output_sizes[-1] # default 512

        # linear layer for feature map reduction
        layers.extend(
            [nn.Flatten(), nn.Linear(output_size, features_size), nn.ReLU()]
        )
        return (nn.ModuleList(layers), maxpooling_layers_positions, cur_signal_length)
    
    def _build_reconstruction_layers(
        self,
        layers_output_sizes:list=[16, 32, 64, 128], 
        output_signal_length:int=512, 
        in_channels:int=1,  
        kernel_size_conv:int=7, 
        kernel_size_maxpool:int=4, 
        n_classes:int=10, 
        padding_size:int=3,
        features_size:int=10)->Tuple[nn.ModuleList, set]:
        """
        Reconstruction from embedings
        """
        
        num_layers = len(layers_output_sizes)
        layers_output_sizes = layers_output_sizes[::-1]
        
        layers = []
        maxunpooling_layers_positions = set()

        # inverse order: linear layer first
        layers.append(
            nn.Linear(
                n_classes,
                layers_output_sizes[0] * output_signal_length #512 default
            )
        )

        
        layers.append(nn.Unflatten(1, (layers_output_sizes[0], output_signal_length)))
        
        counter = 2
        
        for i in range(num_layers):
            
            # cur out channels
            if i == num_layers - 1:
                cur_out_channels = in_channels
            else:
                cur_out_channels = layers_output_sizes[i+1]

            # memorize cur maxunpool position
            maxunpooling_layers_positions.add(counter + 1)
            counter += 4

            # batchnorm->maxunpool->conv1transpose->relu
            layers.extend(
                    [
                    nn.BatchNorm1d(layers_output_sizes[i]),
                    nn.MaxUnpool1d(kernel_size=kernel_size_maxpool),
                    nn.ConvTranspose1d(
                        in_channels=layers_output_sizes[i],
                        out_channels=cur_out_channels, 
                        kernel_size=kernel_size_conv, 
                        padding=padding_size),
                    nn.ReLU()
                    ]
            )
            
        layers.append(nn.Sigmoid())
        
        return nn.ModuleList(layers), maxunpooling_layers_positions
        
    def forward(self, x)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Firstly obtain feature map, then reconstruct the initial image from it.
        
        Indices from maxunpoolings correspond for those from maxpoolings.
        
        """
        maxpooling_indices = []
        
        for i, layer in enumerate(self.feature_layers):
            if i in self.maxpooling_layers_positions:
                x, indices = layer(x)
                maxpooling_indices.append(indices)
            else:
                x = layer(x)
                
        features = x
        
        counter = 0
        maxpooling_indices = maxpooling_indices[::-1]
        
        for i, layer in enumerate(self.reconstruction_layers):
            if i in self.maxunpooling_layers_positions:
                x = layer(x, maxpooling_indices[counter])
                counter += 1
            else:
                x = layer(x)
                
        return x, features