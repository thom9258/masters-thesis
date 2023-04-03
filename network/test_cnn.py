import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from KIN_MUS_parse import KMSession, KIN_MUS_sessions_get, KMSessions2InputsGts
from th_ai import th_csv, th_dataset, th_dataloaderCreate, th_datasetSlice
from th_ai import th_quickPlot, th_datasetPredict, th_mlp
from th_ai import th_regressionModel, th_tinymlp, th_m5


class M5(nn.Module):
    def __init__(self, n_input, n_output, stride, n_channel, verbose=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)
        self.verbose = verbose

    def forward(self, x):
        # x = torch.tensor(x)[None, ...]
        x = x.unsqueeze(0)
        if self.verbose:
            print("Input shape: ", x.shape)

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        if self.verbose:
            print("Conv1 shape: ", x.shape)

        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        if self.verbose:
            print("Conv2 shape: ", x.shape)

        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        if self.verbose:
            print("Conv3 shape: ", x.shape)

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        if self.verbose:
            print("Conv4 shape: ", x.shape)

        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])

        if self.verbose:
            print("AvgPool shape: ", x.shape)
        x = x.permute(0, 2, 1)
        if self.verbose:
            print("Permute shape: ", x.shape)
        x = self.fc1(x)

        self.verbose = False
        return F.log_softmax(x, dim=2)


def main():
    # Tuneable parameters
    model_name = "CNNmodel.pyclass"
    path = "datasets/KIN_MUS_UJI.mat"
    save_model_after_training = True
    use_existing_model = False
    maxepocs = 100
    batchSize = 16
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
    # network_inputLen = inputLen * n_muscles
    # network_outputLen = gtLen * n_angles

    train_set, val_set = dataset.split(0.8)
    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batchSize,
                                                   shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(val_set,
                                                        batch_size=batchSize,
                                                        shuffle=True)

    # TODO: Create a CNN network here
    # network = M5(n_input=network_inputLen,
    #              n_output=network_outputLen,
    #              stride=16,
    #              n_channel=8,
    #              verbose=True,
    #              )

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    sched = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = th_regressionModel()
    if not use_existing_model:
        # Define network parameters
        model.setup(network, optimizer, loss_function, scheduler=sched)

        # Train new model with provided parameters
        model.train(train_dataloader, validation_dataloader,
                    max_epochs=maxepocs,
                    early_stopping=0)
        if save_model_after_training:
            model.save(model_name)

        print(f"Best MSE: {model.best_validation_MSE}")

        th_quickPlot([model.train_MSEs, model.validation_MSEs],
                     ["train", "valid"],
                     axis_labels=["Epoch", "MSE"])
    else:
        model.load(model_name)


if __name__ == "__main__":
    main()
