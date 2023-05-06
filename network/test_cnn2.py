#!/usr/bin/env python3

import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from KIN_MUS_parse import KMSession, KIN_MUS_sessions_get, KMSessions2InputsGts
from th_ai import th_csv, th_dataset, th_dataloaderCreate, th_datasetSlice
from th_ai import th_quickPlot, th_datasetPredict, th_mlp
from th_ai import th_regressionModel, th_tinymlp


def build_cnn_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                    use_batchnorm=True, pool=False):
    '''
    in_channels: input channels to 2d convolution layer
    out_channels: ouput channels to 2d convolution layer
    kernel_size: kernel size (refer torch conv2d)
    stride: stride (refer torch conv2d)
    padding: padding (refer torch conv2d)
    use_batchnorm: enable/disable batchnorm (refer torch BatchNorm2d)
    pool: enable/disable pooling (refer torch MaxPool2D)

    should return the built CNN layer
    '''
    # LeNet structure layers
    layer = nn.Sequential()
    layer.append(nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           padding=padding,
                           stride=stride,
                           kernel_size=(kernel_size, kernel_size)))
    if use_batchnorm:
        layer.append(nn.BatchNorm2d(out_channels))

    layer.append(nn.ReLU())

    if pool:
        layer.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
    return layer


def build_mlp_layer(inputSize, outputSize, useBatchNorm=False):
    layer = nn.Sequential()
    layer.append(nn.Linear(inputSize, outputSize))
    layer.append(nn.ReLU())
    if useBatchNorm:
        layer.append(nn.BatchNorm2d(outputSize))
    return layer


def build_cnn_model(pool=False):
    '''
    pool: enable/disable pooling
    should return built CNN model
    '''
    model = nn.Sequential()
    # Input Layer
    C = [1, 8]

    for i in range(0, len(C)-1):
        layer = build_cnn_layer(C[i], C[i+1], pool=pool)
        model.append(layer)
    if pool:
        lns = 4*4*C[-1]
    else:
        lns = 32*32*C[-1]

    model.append(nn.Flatten())
    model.append(build_mlp_layer(lns, lns))
    model.append(build_mlp_layer(lns, 10))
    # Create network
    return model


def main():
    # Tuneable parameters
    model_name = "CNNmodel.pyclass"
    path = "datasets/KIN_MUS_UJI.mat"
    save_model_after_training = True
    use_existing_model = False
    maxepocs = 100
    batchSize = 8
    inputLen = 4
    gtLen = 1
    n_sessions_in_trainer = 10

    sessions = KIN_MUS_sessions_get(path)
    inputs, gts = KMSessions2InputsGts(sessions, n_sessions_in_trainer, inputLen, gtLen)
    dataset = th_dataset(inputs, gts)

    # NON-Tuneable parameters below
    n_muscles = dataset.input_dims()[1]
    n_angles = dataset.gt_dims()[1]
    print(f"muscles: {n_muscles} muscles with sample length of {inputLen}")
    print(f"angles:  {n_angles} angles with sample length of {gtLen}")
    network_inputLen = inputLen * n_muscles
    network_outputLen = gtLen * n_angles

    train_set, val_set = dataset.split(0.8)
    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batchSize,
                                                   shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(val_set,
                                                        batch_size=batchSize,
                                                        shuffle=True)
    network = build_cnn_model()




if __name__ == "__main__":
    main()
