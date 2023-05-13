#!/usr/bin/env python3

import random
import math
import torch
import numpy as np

from feature_extract import features_extract, slice_session, concat, KMSessions2ClassifierInputsGts
from KIN_MUS_parse import KMSession, KIN_MUS_sessions_get, KMSessions2InputsGts, KMSession2InputsGts

import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    # Tuneable parameters
    path = "datasets/KIN_MUS_UJI.mat"
    inputLen = 50
    outputLen = 3
    n_sessions_in_trainer = 20

    sessions = KIN_MUS_sessions_get(path)
    inputs, gts = KMSessions2ClassifierInputsGts(sessions[:n_sessions_in_trainer], inputLen)
    inputs = features_extract(inputs)
    # Strip a set of array brackets from inputs and gts
    inputs = [inp[0] for inp in inputs]
    gts = [gt[0] for gt in gts]

    examples = 0
    if examples > 0:
        print(f"len inputs: {len(inputs)}")
        print(f"len gts:    {len(gts)}")
    for inp, gt in zip(inputs[0:examples], gts[0:examples]):
        print(f"result: {gt}")
        print(f"input: {inp}")

    model = LinearDiscriminantAnalysis()
    model.fit(np.array(inputs), np.array(gts).ravel())

    print("="*80)
    print("TRAINED THE DAHM MODEL!")
    print("="*80)

    tests = 60
    test_percents = []
    for i in range(n_sessions_in_trainer, n_sessions_in_trainer+tests):
        inputs, gts = KMSessions2ClassifierInputsGts([sessions[i]], inputLen)
        inputs = features_extract(inputs)
        inputs = [inp[0] for inp in inputs]
        gts = [gt[0] for gt in gts]
        correct = []
        for inp, gt in zip(inputs, gts):
            x = np.array(inp).ravel()
            yhat = model.predict([x])
            # print(f"pred={yhat}, gt={gt}")
            if yhat == gt:
                correct.append(1)
            else:
                correct.append(0)
        total = len(correct)
        correct = correct.count(1)
        if total == 0:
            test_percents.append(-1)
        else:
            test_percents.append(correct/total)

    for i, p in enumerate(test_percents):
        print(f"({i}) -> correct/total = {p}")

    cleaned = []
    for p in test_percents:
        if p != -1:
            cleaned.append(p)
    test_percents = cleaned[:50]

    print("correct/total list")
    print(test_percents)
        


if __name__ == "__main__":
    main()
