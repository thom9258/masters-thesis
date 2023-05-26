#!/usr/bin/env python3

import random
import math
import torch
import numpy as np

from feature_extract import features_extract, slice_session, concat, KMSessions2ClassifierInputsGts
from KIN_MUS_parse import KMSession, KIN_MUS_sessions_get, KMSessions2InputsGts, KMSession2InputsGts
from th_ai import th_csv, th_dataset, th_dataloaderCreate, th_datasetSlice
from th_ai import th_quickPlot, th_datasetPredict, th_mlp
from th_ai import th_regressionModel, th_tinymlp

def main():
    # Tuneable parameters
    model_name = "CNNmodel.pyclass"
    path = "datasets/KIN_MUS_UJI.mat"
    save_model_after_training = True
    use_existing_model = False
    maxepocs = 20
    batchSize =  8
    inputLen = 10
    outputLen = 3
    gtLen = 1
    n_sessions_in_trainer = 50

    sessions = KIN_MUS_sessions_get(path)
    inputs, gts = KMSessions2ClassifierInputsGts(sessions[:n_sessions_in_trainer], inputLen)

    inputs = features_extract(inputs)

    examples = 1
    for inp, gt in zip(inputs[0:examples], gts[0:examples]):
        print(f"onehot: {gt}, input: {inp}")

    dataset = th_dataset(inputs, gts)
    # NON-Tuneable parameters below
    train_set, val_set = dataset.split(0.8)
    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batchSize,
                                                   shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(val_set,
                                                        batch_size=batchSize,
                                                        shuffle=True)

    network_inputLen = inputLen * dataset.input_dims()[1]
    print(f"Input Length {network_inputLen}")

    network = th_mlp()
    network.create(inputSize=network_inputLen,
                   outputSize=outputLen,
                   hiddenLayerSize=24,
                   hiddenLayerCount=2,
                   useBatchNorm=False,
                   )

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    sched = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = th_regressionModel()
    model.setup(network, optimizer, loss_function, scheduler=sched)
    # Train new model with provided parameters
    model.train(train_dataloader, validation_dataloader,
                max_epochs=maxepocs,
                early_stopping=0)

    print(f"Best MSE: {model.best_validation_MSE}")
    th_quickPlot([model.train_MSEs, model.validation_MSEs],
                 ["train", "valid"],
                 axis_labels=["Epoch", "Cross Entropy"])

    print("="*80)
    print("="*80)
    print("="*80)
    return

    def datasetPredict(s_inputs, s_gts, trainer):
        gts = []
        preds = []

        angle = 1
        # Predict a regression line for given spliced distribution
        trainer.model.eval()
        for si, sg in zip(s_inputs, s_gts):
            gts.append(sg[0][angle])
            # Generate a prediction using trained model
            si = torch.FloatTensor(np.array([si]))
            res = trainer.predict(si)[0][angle]
            preds.append(res)
        return preds, gts

    inputs, gts = KMSessions2InputsGts([sessions[2]], 1, inputLen, gtLen)
    gts, preds = datasetPredict(inputs, gts, model)

    # Plot ground truth distribution and its predicted counterpart
    th_quickPlot([preds, gts],
                 ["GT", "Prediction"],
                 axis_labels=["GT Angle", "Predicted Angle"])


    inputs, gts = KMSessions2InputsGts([sessions[3]], 1, inputLen, gtLen)
    gts, preds = datasetPredict(inputs, gts, model)

    # Plot ground truth distribution and its predicted counterpart
    th_quickPlot([preds, gts],
                 [f"GT (Angle)", "Prediction"],
                 axis_labels=["Timestep", "Angle [Degrees]"])


if __name__ == "__main__":
    main()
