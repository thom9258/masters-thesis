import torch
from torch import nn
from torch.nn import functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_size, output_size, verbose=False):
        super(SimpleCNN, self).__init__()
        self.verbose = verbose

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size*32, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        if self.verbose:
            print("="*50)
            print("Network Shape:")
            print("input shape: ", x.shape)
        # NOTE: We unsqueeze because our data needs an extra dimension:
        # We want this structure: [batchsz, channels, w, h]
        # so we unsqueeze and permute our input: [batchsz, w, h]
        x = x.unsqueeze(0)
        if self.verbose:
            print("unsqueeze shape: ", x.shape)
        x = x.permute(1, 0, 2, 3)
        if self.verbose:
            print("unsqueeze+permute shape: ", x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        if self.verbose:
            print("conv1 shape: ", x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        if self.verbose:
            print("conv2 shape: ", x.shape)
        # Linear Part
        x = self.flatten(x)
        if self.verbose:
            print("flattened shape: ", x.shape)
        x = self.fc1(x)
        x = self.relu3(x)
        if self.verbose:
            print("linear1 shape: ", x.shape)
        x = self.fc2(x)
        if self.verbose:
            print("linear2 shape: ", x.shape)

        if self.verbose:
            print("output shape: ", x.shape)
            print("="*50)
        self.verbose = False
        return x

class SimpleCNN2(nn.Module):
    def __init__(self, input_size, output_size, verbose=False):
        super(SimpleCNN2, self).__init__()
        self.verbose = verbose

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size*32, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        if self.verbose:
            print("="*50)
            print("Network Shape:")
            print("input shape: ", x.shape)
        # NOTE: We unsqueeze because our data needs an extra dimension:
        # We want this structure: [batchsz, channels, w, h]
        # so we unsqueeze and permute our input: [batchsz, w, h]
        x = x.unsqueeze(0)
        if self.verbose:
            print("unsqueeze shape: ", x.shape)
        x = x.permute(1, 0, 2, 3)
        if self.verbose:
            print("unsqueeze+permute shape: ", x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        if self.verbose:
            print("conv1 shape: ", x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        if self.verbose:
            print("conv2 shape: ", x.shape)
        x = self.conv3(x)
        x = self.relu3(x)
        if self.verbose:
            print("conv3 shape: ", x.shape)
        # Linear Part
        x = self.flatten(x)
        if self.verbose:
            print("flattened shape: ", x.shape)
        x = self.fc1(x)
        x = self.relu3(x)
        if self.verbose:
            print("linear1 shape: ", x.shape)
        x = self.fc2(x)
        x = self.relu4(x)
        if self.verbose:
            print("linear2 shape: ", x.shape)
        x = self.fc3(x)

        if self.verbose:
            print("output shape: ", x.shape)
            print("="*50)
        self.verbose = False
        return x



