import torch
import torch.nn as nn
import torch.nn.functional as F

# from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential
from torch.nn.init import xavier_uniform, xavier_normal


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()
        print(self)


    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions (you will need to add padding). Apply 2x2 Max
        # Pooling to reduce dimensions.
        # If P>N you should implement:
        # (Conv -> ReLU)*N
        # Hint: use loop for len(self.filters) and append the layers you need to the list named 'layers'.
        # Use :
        # if <layer index>%self.pool_every==0:
        #     ...
        # in order to append maxpooling layer in the right places.
        # ====== YOUR CODE: ======
        num_filters = len(self.filters)

        for i in range(num_filters):
            if i % self.pool_every == 0 and i != 0:
                layers.append(nn.MaxPool2d(kernel_size=2))
            
            if i == 0:
              layers.append(nn.Conv2d(in_channels, self.filters[0], kernel_size=(3, 3), padding=1))
              layers.append(nn.ReLU())
            else:
              layers.append(nn.Conv2d(self.filters[i - 1], self.filters[i], kernel_size=(3, 3), padding=1))
              layers.append(nn.ReLU())

        if num_filters >= self.pool_every:
          layers.append(nn.MaxPool2d(kernel_size=2))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w = tuple(self.in_size)
        layers = [] 
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # Hint: use loop for len(self.hidden_dims) and append the layers you need to list named layers.
        # ====== YOUR CODE: ======

        # calculate num of features
        num_of_pools = int(len(self.filters) / self.pool_every)
        dim_in = in_h / (2 ** num_of_pools)
        dim_in = (dim_in ** 2) * self.filters[-1]

        # append layers for FC
        layers.append(nn.Linear(int(dim_in), self.hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input (using self.feature_extractor), flatten your result (using torch.flatten),
      
        # run the classifier on them (using self.classifier) and return class scores.
        # ====== YOUR CODE: ======
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)   
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions (you will need to add padding). Apply 2x2 Max
        # Pooling to reduce dimensions.
        # If P>N you should implement:
        # (Conv -> ReLU)*N
        # Hint: use loop for len(self.filters) and append the layers you need to the list named 'layers'.
        # Use :
        # if <layer index>%self.pool_every==0:
        #     ...
        # in order to append maxpooling layer in the right places.
        # ====== YOUR CODE: ======
        num_filters = len(self.filters)
        layers.append(nn.BatchNorm2d(3, affine=False))
        
        for i in range(num_filters):
            if i % self.pool_every == 0 and i != 0:
                layers.append(nn.MaxPool2d(kernel_size=2))

            if i == 0:
                layers.append(nn.Conv2d(in_channels, self.filters[0], kernel_size=(3, 3), padding=1))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Conv2d(self.filters[i - 1], self.filters[i], kernel_size=(3, 3), padding=1))
                layers.append(nn.ReLU())

        if num_filters >= self.pool_every:
            layers.append(nn.MaxPool2d(kernel_size=2))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w = tuple(self.in_size)
        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # Hint: use loop for len(self.hidden_dims) and append the layers you need to list named layers.
        # ====== YOUR CODE: ======

        # calculate num of features
        num_of_pools = int(len(self.filters) / self.pool_every)
        dim_in = in_h / (2 ** num_of_pools)
        dim_in = (dim_in ** 2) * self.filters[-1]

        # append layers for FC
        layers.append(nn.Linear(int(dim_in), self.hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input (using self.feature_extractor), flatten your result (using torch.flatten),

        # run the classifier on them (using self.classifier) and return class scores.
        # ====== YOUR CODE: ======
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        # ========================
        return out
