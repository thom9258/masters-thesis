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
        self.out = nn.Linear(hlsize, outputlen)
        self.verbose = verbose

    def forward(self, x, h_state):
        # x [batch, time_step, input_size]
        # h_state [n_layers, batch, hidden_size]
        # r_out [batch, time_step, hidden_size]

        if self.verbose:
            print(self.rnn)

        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))

        self.verbose = False
        return torch.stack(outs, dim=1), h_state



def main():
    # https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/403_RNN_regressor.py#L44
    # https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

    # Tuneable parameters
    path = "datasets/KIN_MUS_UJI.mat"
    train_count = 100
    inputLen = 1
    gtLen = 1
    angle_to_keep = 3

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
                print(f"type={type(na)} -> {na}")
                #print(f"type={type(na)}, len={na.size()} -> {na}")
                vbi = False
        for a in gt:
            na = np.array(a[0][angle_to_keep])
            t_cut_gt_angles.append(na)
            if vbg:
                print(f"type={type(na)} -> {na}")
                #print(f"type={type(na)}, len={na.size()} -> {na}")
                vbg = False
        return t_cut_input_muscles, t_cut_gt_angles

    t_input_muscles, t_gt_angles = rnndata_prepare(t_input_muscles, t_gt_angles, verbose=True)
    # print("="*50)
    # print("Processed session to expected format.")

    n_muscles = t_input_muscles[0].size
    n_angles = t_gt_angles[0].size
    print(f"muscles: {n_muscles} muscles with sample length of {inputLen}")
    print(f"angles:  {n_angles} angles with sample length of {gtLen}")

    network = RNN(inputlen=n_muscles, outputlen=n_angles,
                  hlsize=32, hlcount=2,
                  batch_first=True, verbose=True)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # ==========================================================
    # Train the RNN on our session
    # ==========================================================

    print("="*50)
    print("Starting network training!")

    TIME_STEP = 10
    h_state = None      # for initial hidden state

    plt.figure(1, figsize=(12, 5))
    plt.ion()           # continuously plot

    steps = 0
    for x, y in zip(t_input_muscles, t_gt_angles):

        prediction, h_state = network(x, h_state)
        # !! next step is important !!
        h_state = h_state.data        # repack the hidden state, break the connection from last iteration

        loss = loss_function(prediction, y)         # calculate loss
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients

        # plotting
        plt.plot(steps, y, 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.05)
        steps += 1

    plt.ioff()
    plt.show()

    print("Trained the model!")


if __name__ == "__main__":
    main()
