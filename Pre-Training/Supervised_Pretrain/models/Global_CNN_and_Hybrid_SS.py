"""
Fle contains class objects for a global and semi-length agnostic CNN & Hybrid model

Compared to the OG approach, we remove one of the max pooling layers in the 
    encoder with a global pooling operation at the end of the network. 
This global pooling should lead to some nice agnosticity for the model wrt input
    length and feature representations. 
"""

##############################################################################
# IMPORTS
##############################################################################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


##############################################################################
# CONV BLOCK & ENCODER
##############################################################################
def conv_block(in_channels, out_channels, pool_dim, pool):
    """Returns a convolutional block that performs a 3x3 convolution, ReLu
    activation and a variable max pooling

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        pool_dim (int or tuple): Pooling variable for both kernel and stride,
            mainly used to ensure models created can work with smaller/larger
            sequences without reducing dims too far.
        pool (boolean): Whether to use a max pooling layer

    Returns:
        Torch nn module: The torch nn sequntial object with conv, batchnorm, relu
            and maxpool
    """
    if pool == True:
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_dim, stride=pool_dim))
    else:
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    return block


def conv_encoder(in_channels, hidden_channels, pool_dim, pools):
    """Generates a convolutonal based encoder

    Args:
        in_channels (int): The inital number of input channels into the
            encoder
        hidden_channels (int): The number of hidden channels in the convolutional
            procedure
        pool_dim (int or tuple): Pooling stride and kernel variable
        pools (list of booleans): Whether to apply pooling on each convolutional 
            block 
    Returns:
        Torch nn module: The torch encoding model
    """
    return nn.Sequential(
        conv_block(in_channels, hidden_channels, pool_dim, pools[0]),
        conv_block(hidden_channels, hidden_channels, pool_dim, pools[1]),
        conv_block(hidden_channels, hidden_channels, pool_dim, pools[2]),
        conv_block(hidden_channels, hidden_channels, pool_dim, pools[3]),
    )


##############################################################################
# CNN
##############################################################################
class GlobalCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, pool_dim, out_dim):
        """Standard CNN backbone for meta-learning applications

        Args:
            in_channels (int): Number of input channels for the data
            hidden_channels (int): Number of hidden channels used thoughout the
                 main encoder structure
            pool_dim (int or tuple): Pooling stride and kernel variable
            out_dim (int): Number of nodes to output to in final linear layer
        """
        super(GlobalCNN, self).__init__()

        pools = [True, True, True, False]
        self.conv_encoder = conv_encoder(in_channels, hidden_channels, pool_dim, pools)

        self.logits = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(hidden_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(in_features=hidden_channels, out_features=out_dim)
            )

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Trainable Params: {self.params}')

    def forward(self, x, features):
        x = self.conv_encoder(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)

        x = x.view(x.size(0), -1)

        if features:
            return self.logits(x), x
        else:
            return self.logits(x)


##############################################################################
# GLOBAL HYBRID CONVOLuTIONAL-SEQUENTIAL
##############################################################################
class GlobalHybrid(nn.Module):
    def __init__(self,
                in_channels,
                seq_layers,
                seq_type,
                bidirectional,
                hidden_channels,
                pool_dim,
                out_dim):
        """Standardised conv-seq hybrid base learner. Shares a base convolutional
            encoder with the standrdised CNN

        Args:
            in_channels (int): Number of input channels for the data
            seq_layers (int): Number of layers to use in teh sequential part
            seq_type (str): The sequential layer type to use
            bidirectional (boolean): Whether the seq model part should be bidirectional
            hidden_channels (int): Number of hidden channels in teh conv encoder
            pool_dim (int or tuple): MaxPool kernel and stride
            out_dim (int): Number of logits to output to

        Raises:
            ValueError: Error raised if sequential layer type not in ['LSTM',
                'GRU', 'RNN']
        """
        super(GlobalHybrid, self).__init__()

        self.bidirectional = bidirectional
        self.seq_type = seq_type

        # This is the number of output channels * floor_div(n_mels, pool, 4)
        hidden=64
        # Convolutional base encoder
        pools = [True, True, True, False]
        self.conv_encoder = conv_encoder(in_channels, hidden_channels, pool_dim, pools)

        # Make sure value enetered is reasonable
        if seq_type not in ['LSTM', 'GRU', 'RNN']:
            raise ValueError('Seq type not recognised')

        # Generates the sequential layer call
        seq_layer_call = getattr(nn, seq_type)
        self.seq_layers = seq_layer_call(input_size=hidden, hidden_size=hidden,
                        num_layers=seq_layers, bidirectional=bidirectional,
                        batch_first=True)

        # We enforce having a final linear layer with batch norm for ocnvergence
        self.logits = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(hidden, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(in_features=hidden, out_features=out_dim)
            )

        # Count and print the number of trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Num Layers: {seq_layers} -> Trainable Params: {self.params}')

    def many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def forward(self, x, features):
        x = self.conv_encoder(x)

        x = F.adaptive_avg_pool2d(x, output_size=1)

        # (batch, time, freq, channel)
        x = x.transpose(1, -1)
        batch, time = x.size()[:2]

        #(batch, time, channel*freq)
        x = x.reshape(batch, time, -1)

        # Pass through the sequential layers
        if self.seq_type == 'LSTM':
            output, (hn, cn) = self.seq_layers(x)
        else:
            output, hn = self.seq_layers(x)

        forward_output = output[:, :, :self.seq_layers.hidden_size]
        backward_output = output[:, :, self.seq_layers.hidden_size:]


        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i)
        # AKA A skip connection between inputs and outputs is used
        if self.bidirectional:
            x = forward_output + backward_output + x
        else:
            x = forward_output + x

        x = self.many_to_one(x, x.shape[-2])

        if features:
            return self.logits(x), x
        else:
            return self.logits(x)


##############################################################################
# DATA PASS EXAMPLE
##############################################################################
"""
data = torch.rand(10, 1, 128, 1000)
model = GlobalHybrid(1, 64, 3, 5)
out = model.forward(data)
print(out.shape)
"""
"""
data = torch.rand(10, 1, 128, 32)
model = GlobalHybrid(in_channels=1,
                seq_layers=1,
                seq_type='RNN',
                bidirectional=False,
                hidden_channels=64,
                pool_dim=(3,3),
                out_dim=5)

out = model.forward(data)
print(out.shape)
"""

