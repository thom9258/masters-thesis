import random
import torch
import numpy as np

from KIN_MUS_parse import KMSession, kin_mus_parse
from th_ai import th_csv, th_dataset, th_dataloaderCreate, th_datasetSlice
from th_ai import th_quickPlot, th_datasetPredict, th_mlp
from th_ai import th_regressionModel, th_tinymlp, th_m5


def sessionSlice(session):
    inputLen = 10
    outputLen = 1
    inputs = []
    gts = []
    maxlen = len(session.time)

    # Slice up session into prediction sets
    for i in range(maxlen - inputLen):

        # Extract inputs for an accociated prediction
        inp = []
        for j in range(inputLen):
            ofs = i + j
            inp.append(session.muscles[ofs])
            # print(f"I-offset = {ofs}, for point {j} ")
        inputs.append(np.array(inp).flatten())

        # Extract outputs for an accociated prediction
        gt = []
        for j in range(outputLen):
            ofs = i + j + inputLen
            gt.append(session.angles[ofs])
            # print(f"O-offset = {ofs}, for point {j} ")
        gts.append(np.array(gt).flatten())

    return inputs, gts


def concat(a, b):
    out = []
    for v in a:
        out.append(v)
    for v in b:
        out.append(v)
    return out


def distribution_create(sessions):
    inputs, gts = [], []

    # i, g = sessionSlice(sessions[0])
    # inputs.append(i)
    # gts.append(g)
    for s in sessions:
        i, g = sessionSlice(s)
        inputs = concat(inputs, i)
        gts = concat(gts, g)
        # inputs.append(i)
        # gts.append(g)

    return inputs, gts


def main():
    # Tuneable parameters
    model_name = "model"
    path = "datasets/KIN_MUS_UJI.mat"
    save_model_after_training = True
    use_existing_model = False
    inpsize = 10*7
    outsize = 1*18
    maxepocs = 20
    batchSize = 16

    network = th_mlp()
    network.create(inputSize=inpsize,
                   outputSize=outsize,
                   hiddenLayerSize=100,
                   hiddenLayerCount=2,
                   useBatchNorm=False)

    network = th_tinymlp(inpsize, outsize)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    #optimizer = None
    #sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    sched = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    sessions = kin_mus_parse(path)
    inputs, gts = distribution_create(sessions[0:50])
    dataset = th_dataset(inputs, gts)
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
                     axis_labels=["batch", "MSE"])
    else:
        model.load(model_name)


if __name__ == "__main__":
    main()
