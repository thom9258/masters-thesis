import random
import torch
import numpy as np

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
    maxepocs = 100
    batchSize = 16
    inputLen = 10
    gtLen = 1
    n_sessions_in_trainer = 10

    sessions = KIN_MUS_sessions_get(path)
    inputs, gts = KMSessions2InputsGts(sessions, n_sessions_in_trainer, inputLen, gtLen)

    print(f"INPUTS size = {inputs[0].shape}")
    print(f"GTS size    = {gts[0].shape}")

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

    network = th_mlp()
    network.create(inputSize=network_inputLen,
                   outputSize=network_outputLen,
                   hiddenLayerSize=128,
                   hiddenLayerCount=4,
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

    print("="*80)
    print("="*80)
    print("="*80)

    inputs, gts = KMSession2InputsGts(sessions[0], inputLen, gtLen)

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

    gts, preds = datasetPredict(inputs, gts, model)

    # Plot ground truth distribution and its predicted counterpart
    th_quickPlot([gts, preds],
                 ["GT", "Prediction"],
                 axis_labels=["GT Angle", "Predicted Angle"])




if __name__ == "__main__":
    main()
