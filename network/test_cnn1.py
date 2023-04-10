import random
import sys
import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
import numpy as np

from KIN_MUS_parse import KMSession, KIN_MUS_sessions_get, KMSessions2InputsGts
from th_ai import th_csv, th_dataset, th_dataloaderCreate, th_datasetSlice
from th_ai import th_quickPlot, th_mlp


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


class regressionModel:
    def __init__(self):
        pass

    def setup(self, model, optimizer, loss_function, scheduler=None):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.train_losses, self.train_MSEs = [], []
        self.validation_losses, self.validation_MSEs = [], []
        self.best_validation_MSE = sys.maxsize
        self.best_validation_MSE_epoch = 0

    def predict(self, inp):
        return list(self.model(inp).detach().numpy())

    def epoch(self, data_loader, is_training):
        losses, squared_errors = [], []
        # Iterate over the DataLoader for training data
        for inputs, targets in data_loader:
            inputs, targets = inputs.float(), targets.float()
            # print(f"INPUTS {inputs}")
            # print(f"TARGETS {targets}")
            # Perform forward pass
            outputs = self.model(inputs)
            # Compute loss
            loss = self.loss_function(outputs, targets)
            losses.append(loss.item())

            if is_training:
                # Zero the gradients
                self.optimizer.zero_grad()
                # Perform backward pass
                loss.backward()
                # Perform optimization
                self.optimizer.step()
                # Use a Learning Rate scheduler if provided
                if self.scheduler is not None:
                    self.scheduler.step(losses[-1])

            # ========================
            # Bookkeeping for graphs:
            squared_errors.append(
                ((outputs - targets)*(outputs - targets)).sum().data)

        return squared_errors, losses

    def train(self, loader_train, loader_valid, max_epochs, early_stopping=0):
        if early_stopping <= 0:
            early_stopping = max_epochs+1

        t = tqdm(range(max_epochs))
        for epoch_count in t:
            # Test Set epoch
            sqrd_errors, losses = self.epoch(loader_train, is_training=True)
            self.train_losses.append(np.mean(losses))
            self.train_MSEs.append(np.mean(sqrd_errors))

            # Validation Set epoch
            sqrd_errors, losses = self.epoch(loader_valid, is_training=False)
            self.validation_losses.append(np.mean(losses))
            self.validation_MSEs.append(np.mean(sqrd_errors))

            desc = "train/test MSE: {:.2f}/{:.2f}".format(
                self.train_MSEs[-1],
                self.validation_MSEs[-1])
            t.set_description(desc)

            if self.validation_MSEs[-1] < self.best_validation_MSE:
                self.best_validation_MSE = self.validation_MSEs[-1]
                self.best_validation_MSE_epoch = epoch_count

            if epoch_count > self.best_validation_MSE_epoch + early_stopping:
                break
# regressionModel


def main():
    # Tuneable parameters
    path = "datasets/KIN_MUS_UJI.mat"
    maxepocs = 100
    batchSize = 8
    inputLen = 10
    gtLen = 1
    n_sessions_in_trainer = 1
    angle = 12

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

    network = SimpleCNN(input_size=network_inputLen,
                        output_size=network_outputLen,
                        verbose=True)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    sched = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = regressionModel()
    model.setup(network, optimizer, loss_function, scheduler=sched)
    model.train(train_dataloader, validation_dataloader,
                max_epochs=maxepocs,
                early_stopping=0)

    print("Trained the model!")
    print(f"Best MSE: {model.best_validation_MSE}")
    th_quickPlot([model.train_MSEs, model.validation_MSEs],
                 ["train", "valid"],
                 axis_labels=["Epoch", "MSE"])

    print("="*80)
    print("="*80)
    print("="*80)

    inputs, gts = KMSessions2InputsGts([sessions[0]], 1, inputLen, gtLen)

    def datasetPredict(s_inputs, s_gts, trainer):
        gts = []
        preds = []
        # print(f"Given length = {len(s_inputs)}")
        # Predict a regression line for given spliced distribution
        trainer.model.eval()
        for si, sg in zip(s_inputs, s_gts):
            gts.append(sg[0][angle])
            # Generate a prediction using trained model
            si = torch.FloatTensor(np.array([si]))
            res = trainer.predict(si)[0][angle]
            preds.append(res)
        return preds, gts

    gts, preds = datasetPredict(inputs, gts, model)

    # Plot ground truth distribution and its predicted counterpart
    th_quickPlot([gts, preds],
                 [f"GT (Angle={angle})", "Prediction"],
                 axis_labels=["Timestep", "Angle [Degrees]"])


if __name__ == "__main__":
    main()
