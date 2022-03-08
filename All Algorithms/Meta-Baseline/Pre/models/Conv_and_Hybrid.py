"""
File contains all necessary functions and classes to create either a purely
    convolutional modle or a variety of convolutional-sequential hybrids. These
    models share identical convolutional encoder sections and only measurably 
    differ in the inclusion/ or lack of a sequential model component.

Functions/Classes:
    -> Floor division function we use to calculate logit layer size for CNN 
    -> Smaller conv block used to create the encodeing structure:
        - Each with; Conv2d, BatchNorm2d, ReLU, MaxPool2d
        - Conv2d has kernel=3 and padding=1 fixed
        - Stride and kernel for MaxPool is controlled by pool_dim variable
    -> Standardised Convolutional encoder(shared between pure conv and hybrids)
    -> Purely convolutional model class 
    -> Hybrid model class with various sequential modle types supoorted,
        LSTM/GRU/RNN

Like all base learners created and considered, the option for encoding without
    classification is included(added as out_dim just being different from num_classes) 
    along with max pool dimensionality control, to make sure models are suitable 
    for the length of dataset we want to work with, time dimensionality wise. 

The 'Standard' prefix to the models is to confirm that they use a standardised
    backbone in experiment logs.
"""

##############################################################################
# IMPORTS
##############################################################################
import torch
import numpy as np
import torch.nn as nn

##############################################################################
# Other Functions
##############################################################################
def floor_power(num, divisor, power):
    """Performs what we call a floor power, a recursive fixed division process 
        with a flooring between each time

    Args:
        num (int or float):The original number to divide from
        divisor (int or float): The actual divisor for the number
        power (int): How many times we apply this divide and then floor

    Returns:
        int: The numerical result of the floor division process
    """
    for _ in range(power):
        num = np.floor(num/divisor)
    return num

##############################################################################
# CONV BLOCK & ENCODER
##############################################################################
def conv_block(in_channels, out_channels, pool_dim):
    """Returns a convolutional block that performs a 3x3 convolution, ReLu 
    activation and a variable max pooling

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        pool_dim (int or tuple): Pooling variable for both kernel and stride,
            mainly used to ensure models created can work with smaller/larger
            sequences without reducing dims too far.

    Returns:
        Torch nn module: The torch nn seuqntial object with conv, batchnorm, relu
            and maxpool
    """
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=pool_dim, stride=pool_dim))
    return block


def conv_encoder(in_channels, hidden_channels, pool_dim):
    """Generates a convolutonal based encoder

    Args:
        in_channels (int): The inital number of input channels into the
            encoder
        hidden_channels (int): The number of hidden channels in the convolutional
            procedure
        pool_dim (int or tuple): Pooling stride and kernel variable

    Returns:
        Torch nn module: The torch encoding model
    """
    return nn.Sequential(
        conv_block(in_channels, hidden_channels, pool_dim),
        conv_block(hidden_channels, hidden_channels, pool_dim),
        conv_block(hidden_channels, hidden_channels, pool_dim),
        conv_block(hidden_channels, hidden_channels, pool_dim),
    )

##############################################################################
# CNN 
##############################################################################
class StandardCNN(nn.Module):
    def __init__(self, in_channels, trial_shape, hidden_channels, pool_dim, out_dim):
        """Standard CNN backbone for meta-learning applications

        Args:
            in_channels (int): Number of input channels for the data
            trial_shape (tuple or array)): An example data sample shape array/tuple,
                used to work out the input to the final linear layer
            hidden_channels (int): Number of hidden channels used thoughout the
                 main encoder structure 
            pool_dim (int or tuple): Pooling stride and kernel variable
            out_dim (int): Number of nodes to output to in final linear layer
            logits (int): Number of nodes ot ouput to for strict softmax classification
        """
        super(StandardCNN, self).__init__()
        self.conv_encoder = conv_encoder(in_channels, hidden_channels, pool_dim)

        # Caluclates how many nodes needed to collapse from conv layer
        num_logits = int(64 * floor_power(trial_shape[2], pool_dim[0], 4) * floor_power(trial_shape[3], pool_dim[1], 4))

        self.logits = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(num_logits, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(in_features=num_logits, out_features=out_dim)
            )

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Trainable Params: {self.params}')

    def forward(self, x, features):
        x = self.conv_encoder(x)
        x = x.view(x.size(0), -1)

        if features:
            return self.logits(x), x
        else:
            return self.logits(x)

##############################################################################
# HYBRID CONVOLUTIONAL-SEQUENTIAL 
##############################################################################
class StandardHybrid(nn.Module):
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
        super(StandardHybrid, self).__init__()

        self.bidirectional = bidirectional
        self.seq_type = seq_type

        # This is the number of output channels * floor_div(n_mels, pool, 4)
        hidden=64
        # Convolutional base encoder
        self.conv_encoder = conv_encoder(in_channels, hidden_channels, pool_dim)

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
# EXAMPLE PASS THORUGH
##############################################################################
"""
data = torch.rand(10, 1, 128, 157)
model = StandardHybrid(in_channels=1,
                seq_layers=2,
                seq_type='RNN',
                bidirectional=False,
                hidden_channels=64,
                pool_dim=3,
                out_dim=5)

out = model.forward(data)
print(out.shape)


model = StandardCNN(1, data.shape, 64, 3, 5)
out = model.forward(data)
print(out.shape)
# At time dim <=80, we have to use pool_dim=(3,2)
"""