#!/usr/bin/env python3

import csv
import sys
import random
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# pickle - save python objects
try:
    import cPickle as pickle
except:
    import pickle


if __name__ == "__main__":
    print("ERROR: running th_ai.py as main is not adviced.")


class th_tinymlp(nn.Module):
    def name(self):
        return "th_tinymlp"

    def __init__(self, inp, outp):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inp, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, outp)
        )

    def forward(self, x):
        return self.network(x)


class th_mlp(nn.Module):
    """
    A MLP (multi-layer perceptron) model of a network
    https://colab.research.google.com/drive/1O52PSINsQj71ZXLDSeRMOZp_EA-J0wOa#scrollTo=M9yuUlok1Xwn
    """

    def name(self):
        return "th_mlp"

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential()
        self.is_ok = False

    def create(self, inputSize, outputSize, hiddenLayerSize=50,
               hiddenLayerCount=1, useBatchNorm=False,
               batchNorm=nn.BatchNorm1d, activationFn=nn.ReLU()):

        # Input Layer
        self.network.append(nn.Flatten())
        self.network.append(nn.Linear(inputSize, hiddenLayerSize))

        # Hidden Layer(s)
        for i in range(0, hiddenLayerCount):
            self.network.append(activationFn)
            self.network.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
            if useBatchNorm:
                self.network.append(batchNorm(hiddenLayerSize))

        # Output Layer
        self.network.append(activationFn)
        self.network.append(nn.Linear(hiddenLayerSize, outputSize))
        if useBatchNorm:
            self.network.append(batchNorm(outputSize))
        self.is_ok = True

    def forward(self, x):
        return self.network(x)

    def parameterCount(self):
        return np.sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def printnn(self):
        print(f"Parameter Count = {self.parameterCount()}")
        print(self.network)
# class th_mlp

class th_regressionModel:
    def __init__(self):
        self.slnametype = ".pyclass"

    def setup(self, model, optimizer, loss_function, scheduler=None):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.train_losses, self.train_MSEs = [], []
        self.validation_losses, self.validation_MSEs = [], []
        self.best_validation_MSE = sys.maxsize
        self.best_validation_MSE_epoch = 0

    def save(self, filename):
        fname = filename+self.slnametype
        with open(filename, 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved model as [{fname}]")

    def load(self, filename):
        fname = filename+self.slnametype
        with open(fname, 'rb') as handle:
            self.__dict__ = pickle.load(handle)
        print(f"Loaded model from [{fname}]")

    def predict(self, inp):
        if type(inp) is list:
            inp = torch.FloatTensor(inp)
        return list(self.model(inp).detach().numpy())

    def epoch(self, data_loader, is_training):
        losses, squared_errors = [], []
        # Iterate over the DataLoader for training data
        for i, (inputs, targets) in enumerate(data_loader, 0):
            inputs, targets = inputs.float(), targets.float()
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
# th_regressionModel


class OLD_th_regressionModel:
    """
    th_AI - Trainer module.
    """

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

    def save(self, filename):
        fname = filename+'.pytxt'
        with open(fname, 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved model as [{fname}]")

    def load(self, filename):
        fname = filename+'.pytxt'
        with open(fname, 'rb') as handle:
            self.__dict__ = pickle.load(handle)
        print(f"Loaded model from [{fname}]")

    def predict(self, inp):
        if type(inp) is list:
            inp = torch.FloatTensor(inp)
        return list(self.model(inp).detach().numpy())

    def epoch(self, data_loader, is_training):
        losses, squared_errors = [], []
        # Iterate over the DataLoader for training data
        for i, (inputs, targets) in enumerate(data_loader, 0):
            inputs, targets = inputs.float(), targets.float()
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
            if is_training:
                self.train_losses.append(np.mean(losses))
                self.train_MSEs.append(np.mean(squared_errors))
            else:
                self.validation_losses.append(np.mean(losses))
                self.validation_MSEs.append(np.mean(squared_errors))

    def train(self, loader_train, loader_valid, max_epochs, early_stopping=0):
        if early_stopping <= 0:
            early_stopping = max_epochs+1

        t = tqdm(range(max_epochs))
        for epoch_count in t:
            self.epoch(loader_train, is_training=True)
            self.epoch(loader_valid, is_training=False)
            desc = "train/test MSE: {:.2f}/{:.2f}".format(
                self.train_MSEs[-1],
                self.validation_MSEs[-1])
            t.set_description(desc)

            if self.validation_MSEs[-1] < self.best_validation_MSE:
                self.best_validation_MSE = self.validation_MSEs[-1]
                self.best_validation_MSE_epoch = epoch_count

            if epoch_count > self.best_validation_MSE_epoch + early_stopping:
                break
# th_regressionModel


class th_csv:
    """
    th_csv - csv datacontainer.
    A Shitty version of Pandas dataframes.
    """
    def th_csv(this):
        this.data = []

    def load(this, path):
        this.data = []
        if not Path(path).exists:
            print(f"Path [{path}] Does not exist!")
            return
        file = open(path)
        reader = csv.reader(file)
        for row in reader:
            this.data.append(row)

    def isSquare(this):
        # Check if data has varying row lengths (missing data)
        max_len = max([len(x) for x in this.data])
        for row in this.data:
            if len(row) is not max_len:
                return False
        return True

    def hasNoNonValues(this):
        return True

    def isComplete(this):
        ok = this.isSquare()
        if ok is False:
            return ok
        ok = this.hasNoNonValues()
        if ok is False:
            return ok
        return ok

    def getRow(this, idx):
        out = []
        for i, row in enumerate(this.data):
            if i is idx:
                out.append(row)
        return out[0]

    def removeRow(this, rem):
        ndata = []
        for i, row in enumerate(this.data):
            if i is not rem:
                ndata.append(row)
        this.data = ndata

    def getColumn(this, idx):
        out = []
        for row in this.data:
            newrow = []
            for i, val in enumerate(row):
                if i is idx:
                    newrow.append(val)
            out.append(newrow)
        return out

    def removeColumn(this, rem):
        new = []
        for row in this.data:
            newrow = []
            for i, val in enumerate(row):
                if i is not rem:
                    newrow.append(val)
            new.append(newrow)
        this.data = new

    def getHead(this, n=5):
        return this.data[:n]

    def printHead(this, n=5):
        head = this.getHead(n)
        for row in head:
            print(row)

    def shape(this):
        return len(this.data), len(this.data[0])

    def floatify(this):
        new = []
        for row in this.data:
            newrow = []
            for val in row:
                newrow.append(float(val))
            new.append(newrow)
        this.data = new


class th_dataset(torch.utils.data.Dataset):
    def __init__(self, data, gts):
        print(f"inplen/gtslen = {len(data)}/{len(gts)}")
        assert len(data) != 0 or len(gts) != 0
        assert len(data) == len(gts)
        self.data = torch.FloatTensor(np.array(data))
        self.gts = torch.FloatTensor(np.array(gts))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.gts[idx]

    def len(self):
        return len(self.data)

    def input_dims(self):
        return [len(self.data[0]), len(self.data[0][0])]

    def gt_dims(self):
        return [len(self.gts[0]), len(self.gts[0][0])]

    def split(self, percent):
        trainsize = int(self.len()*percent)
        valsize = int(self.len() - trainsize)
        train_set, val_set = torch.utils.data.random_split(self, [trainsize, valsize])
        return train_set, val_set
# th_dataset


def th_quickPlot(data, data_labels, axis_labels=["x", "y"]):
    plt.figure()
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    for i in range(0, len(data)):
        plt.plot(data[i], label=data_labels[i])
    plt.legend()
    plt.grid()
    plt.show()


def th_datasetSlice(distribution, inputLen, outputLen, outputOffset):
    inputs = []
    gts = []
    dlen = len(distribution)

    # Slice up distribution into prediction sets
    for i in range(0, dlen - inputLen - outputLen - outputOffset):
        inp = []
        gt = []
        # Extract inputs for an accociated prediction
        for j in range(0, inputLen):
            inp.append(distribution[i + j][0])
            # print(f"I-offset = {i + j}")

        # Extract outputs for an accociated prediction
        for j in range(0, outputLen):
            gt.append(distribution[inputLen + outputOffset + i + j][0])
            # print(f"O-offset = {inputLen + outputOffset + i + j}")

        # Add prediction set to sliced prediction sets
        inputs.append(inp)
        gts.append(gt)

    return inputs, gts


def th_dataloaderCreate(distribution, inputLen, outputLen, outputOffset, nSets, batchSize):
    inputs = []
    gts = []

    # Slice up distribution to get all possible prediction sets
    s_inputs, s_gts = th_datasetSlice(distribution,
                                      inputLen,
                                      outputLen,
                                      outputOffset=outputOffset)
    for _ in range(0, nSets):
        # Extract random set from prediction sets
        i = random.randint(0, len(s_inputs) - 1)

        # Add random set to dataset
        inputs.append(s_inputs[i])
        gts.append(s_gts[i])

    # Create dataset and dataloader
    dataset = th_dataset(inputs, gts)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batchSize,
                                       shuffle=True)


def th_datasetPredict(s_inputs, s_gts, trainer):
    gts = []
    preds = []

    # Predict a regression line for given spliced distribution
    trainer.model.eval()
    for si, sg in zip(s_inputs, s_gts):
        gts.append(sg)
        # Generate a prediction using trained model
        preds.append(trainer.predict(torch.FloatTensor(si)))
    return preds, gts


class th_argumentation:
    def ShiftLinear(distribution, shift):
        out = []
        for x in distribution:
            out.append(x+shift)
        return out

    def ShiftScaled(distribution, mult):
        out = []
        for x in distribution:
            out.append(x*mult)
        return out

    def GaussNoise(distribution, stdDeviation):
        out = []
        for x in distribution:
            out.append(float(np.random.normal(x, stdDeviation, 1)))
        return out

    def rand(self, distribution):
        x = random.uniform(-4, 4)
        distribution = self.ShiftLinear(distribution, x)
        x = random.uniform(0.9, 1.1)
        distribution = self.ShiftScaled(distribution, x)
        x = random.uniform(-0.5, 0.5)
        distribution = self.GaussNoise(distribution, x)
        return distribution

# th_argumentation
