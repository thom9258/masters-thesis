import random
import torch
import numpy as np

from KIN_MUS_parse import KMSession, KIN_MUS_dataset_parse, KMSessionsAutoPrepare, KMconcat
from th_ai import th_csv, th_dataset, th_dataloaderCreate, th_datasetSlice
from th_ai import th_quickPlot, th_datasetPredict, th_mlp
from th_ai import th_regressionModel, th_tinymlp, th_m5


def subsetFromIndices(arr, indices):
    return [arr[i] for i in indices]


def main():
    # Tuneable parameters
    model_name = "model"
    path = "datasets/KIN_MUS_UJI.mat"
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

    print(f"{n_muscles} muscles -> {n_angles} angles")
    print(f"Angle indices: {angle_indices}")

    network = th_tinymlp(network_inputLen, network_outputLen)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    # optimizer = None
    # sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    sched = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    sessions = KIN_MUS_dataset_parse(path)
    # inputs, gts = KMSessionsAutoPrepare(sessions[:10], inputLen, gtLen)
    sessions = sessions[:10]
    inputs, gts = [], []
    for s in sessions:
        # Extract session data
        i, g = s.slice(inputLen, gtLen)
        inputs = KMconcat(inputs, i)
        gts = KMconcat(gts, g)

    # Cut off unwanted angles from gtlen
    cut_gts = []
    for g in gts:
        cut_gts.append(subsetFromIndices(g, angle_indices))
    gts = cut_gts
    print(f"keeps angles {angle_indices}")

    print(f"input len = {len(inputs[0])}")
    print(f"gt len={len(gts[0])}")

    dataset = th_dataset(inputs, gts)
    print("="*80)

    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batchSize,
                                                   shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(dataset,
                                                        batch_size=batchSize,
                                                        shuffle=True)

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
                     axis_labels=["Batch", "MSE"])
    else:
        model.load(model_name)


if __name__ == "__main__":
    main()
