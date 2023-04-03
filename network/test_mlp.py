import random
import torch
import numpy as np

from KIN_MUS_parse import KMSession, KIN_MUS_dataset_parse, KMSessionsAutoPrepare, KMSubsetDataset
from th_ai import th_csv, th_dataset, th_dataloaderCreate, th_datasetSlice
from th_ai import th_quickPlot, th_datasetPredict, th_mlp
from th_ai import th_regressionModel, th_tinymlp, th_m5


def main():
    # Tuneable parameters
    model_name = "model.pyclass"
    path = "datasets/KIN_MUS_UJI.mat"
    n_sessions = 100
    save_model_after_training = True
    use_existing_model = False
    maxepocs = 100
    batchSize = 8
    inputLen = 16
    gtLen = 1
    n_muscles = 7
    max_angles = 18
    # Create a mask of angles
    angle_indices = [5]
    # angle_indices = [0, 1, 2]
    assert len(angle_indices) < max_angles
    n_angles = len(angle_indices)
    network_inputLen = inputLen * n_muscles
    network_outputLen = gtLen * n_angles

    print(f"Network Inp/Outp = {network_inputLen} / {network_outputLen}")
    print(f"Using the first {n_sessions} sessions for training")
    print(f"{n_muscles} muscles -> {n_angles} angles")
    print(f"Angle indices: {angle_indices}")

    dataset = KMSubsetDataset(path, inputLen, gtLen, angle_indices, n_sessions=n_sessions)
    train_set, val_set = dataset.split(0.8)

    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batchSize,
                                                   shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(val_set,
                                                        batch_size=batchSize,
                                                        shuffle=True)

    network = th_mlp()
    network.create(inputSize=network_inputLen,
                   outputSize=network_outputLen,
                   hiddenLayerSize=2*64,
                   hiddenLayerCount=8,
                   useBatchNorm=False,
                   )

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
