#!/usr/bin/env python3

import random
import sys
import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from KIN_MUS_parse import KMSession, KIN_MUS_sessions_get, KMSession2InputsGts
from th_ai import th_csv, th_dataset, th_dataloaderCreate, th_datasetSlice
from th_ai import th_quickPlot, th_datasetPredict, th_mlp



class RNN(nn.Module):
    def __init__(self, inputlen, outputlen, hlsize, hlcount,
                 batch_first=True, verbose=False):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=inputlen,
            hidden_size=hlsize,
            num_layers=hlcount,
            # Place batch size as first dimension
            # [batch, time_step, input_size]
            batch_first=True,
        )
        print(f"input len = {inputlen}")
        print(f"output len = {outputlen}")
        print(f"hidden layer size = {hlsize}")
        print(f"hidden layer count = {hlcount}")
        self.out = nn.Linear(hlsize, outputlen)
        self.verbose = verbose

    def forward(self, x, h_state):
        # x [batch, time_step, input_size]
        # h_state [n_layers, batch, hidden_size]
        # r_out [batch, time_step, hidden_size]

        x = torch.FloatTensor(x)
        if self.verbose:
            print(self.rnn)
            #print(f"Input = {x}")
            print(f"Input Dims = {x.shape}")
        x = x.unsqueeze(0)
        if self.verbose:
            print(f"Unsqueezed = {x.shape}")

        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :][0]))
            # outs.append(self.out(r_out[:, time_step]))

        self.verbose = False
        outs = torch.stack(outs, dim=1)[0]
        return outs, h_state

def trainRNN(train_dataloader, network, loss_function, optimizer):
    h_state = None      
    predictions = []
    gts = []
    for x, y in train_dataloader:
        x, y = x.float(), y.float()
        prediction, h_state = network(x, h_state)
        # print(f"GT: {y}")
        #print(f"Predicted: {prediction}")
        # !! next step is important !!
        # repack the hidden state, break the connection from last iteration
        h_state = h_state.data

        loss = loss_function(prediction, y)         # calculate loss
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients
        # print("="*50)
        # print(f"GT: (shape={y.numpy().shape}) {y.numpy()}")
        # print(f"Predicted: (shape={prediction.detach().numpy().shape}) {prediction.detach().numpy()}")
        # print("="*50)
        for g,p in zip(y.numpy(), prediction.detach().numpy().flatten()):
            predictions.append(p)
            gts.append(g)
    return gts, predictions


def main():
    # https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/403_RNN_regressor.py#L44
    # https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
    # https://www.analyticsvidhya.com/blog/2021/07/understanding-rnn-step-by-step-with-pytorch/

    # Tuneable parameters
    path = "datasets/KIN_MUS_UJI.mat"
    train_count = 100
    batchSize = 8
    inputLen = 1
    gtLen = 1
    angle_to_keep = 12

    hlsize=128
    hlcount=4
    epochs = 100

    # NON-Tuneable parameters below
    sessions = KIN_MUS_sessions_get(path)

    trainsession = sessions[1]
    t_input_muscles, t_gt_angles = KMSession2InputsGts(trainsession,
                                                       inputLen,
                                                       gtLen,
                                                       verbose=True)

    def rnndata_prepare(inp, gt, verbose=False):
        t_cut_input_muscles = []
        t_cut_gt_angles = []
        vbi = verbose
        vbg = verbose
        for a in inp:
            na = a[0]
            t_cut_input_muscles.append(na)
            if vbi:
                # print(f"type={type(na)} -> {na}")
                vbi = False
        for a in gt:
            na = np.array(a[0][angle_to_keep])
            t_cut_gt_angles.append(na)
            if vbg:
                # print(f"type={type(na)} -> {na}")
                vbg = False
        return t_cut_input_muscles, t_cut_gt_angles

    t_input_muscles, t_gt_angles = rnndata_prepare(t_input_muscles, t_gt_angles, verbose=True)
    dataset = th_dataset(t_input_muscles, t_gt_angles)
    train_set, val_set = dataset.split(0.8)
    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batchSize,
                                                   shuffle=False)
    validation_dataloader = torch.utils.data.DataLoader(val_set,
                                                        batch_size=batchSize,
                                                        shuffle=False)


    n_muscles = t_input_muscles[0].size
    n_angles = t_gt_angles[0].size
    print(f"muscles: {n_muscles} muscles with sample length of {inputLen}")
    print(f"angles:  {n_angles} angles with sample length of {gtLen}")

    network = RNN(inputlen=n_muscles, outputlen=n_angles,
                  hlsize=hlsize, hlcount=hlcount,
                  batch_first=True, verbose=True)

    loss_function = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    # optimizer = torch.optim.Adam(network.parameters(), lr=0.6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # ==========================================================
    # Train the RNN on our session
    # ==========================================================

    for _ in range(epochs):
        _, _ = trainRNN(train_dataloader, network, loss_function, optimizer)
           
    # Plot ground truth distribution and its predicted counterpart
    gts, predictions = trainRNN(train_dataloader, network, loss_function, optimizer)
    th_quickPlot([gts, predictions],
                 [f"GT angle ({angle_to_keep})",
                  "Prediction"],
                 axis_labels=["Timestep", "Angle [Degrees]"])

    # th_quickPlot([gts],
    #              [f"GT (Angle={angle_to_keep})"],
    #              axis_labels=["Timestep", "Angle [Degrees]"])
    

if __name__ == "__main__":
    main()
