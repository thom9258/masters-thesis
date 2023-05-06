from scipy.io import loadmat
import numpy as np
import torch

from th_ai import th_dataset, th_dataloaderCreate, th_datasetSlice


class KMSession:
    def __init__(self, subject_id, adl, phase, time, angles, muscles):
        """
        Dataset found at:
        https://zenodo.org/record/3337890#.ZApkdYDMJhF
        Managing .mat files:
        https://stackoverflow.com/questions/42424338/how-to-load-mat-files-in-python-and-access-columns-individually
        adl is the action a person did, the phase of movement is the action's
        process given by index:
            1 = reaching
            2 = manipulation
            3 = releasing
        """
        self.ok = False
        for a, m in zip(angles, muscles):
            if len(a) != 18 or len(m) != 7:
                print(f"WARNING: Invalid session, angle/muscles is [{len(a)}/{len(m)}] and NOT [18/7]")
                return
        self.subject_id = subject_id
        self.adl = adl
        self.time = time
        self.phase = phase
        self.angles = angles
        self.muscles = muscles
        self.ok = True

    def as_str(self):
        out = ""
        out += f"\t[id: {self.subject_id}  "
        out += f"adl: {self.adl}  "
        out += f"time: {self.time[0]} -> {self.time[-1]}]"
        return out

    def slice_sequence_into_InputsGts(self, inputLen, gtLen):
        inputs = []
        gts = []
        maxlen = len(self.time)
        # Slice up session into prediction sets
        for i in range(maxlen - inputLen):
            # Extract inputs for an accociated prediction
            inp = []
            for j in range(inputLen):
                ofs = i + j
                inp.append(self.muscles[ofs])
                # print(f"I-offset = {ofs}, for point {j} ")
            inputs.append(np.array(inp))
            # Extract outputs ground truth for an accociated prediction
            gt = []
            for j in range(gtLen):
                ofs = i + j + inputLen
                gt.append(self.angles[ofs])
                # print(f"O-offset = {ofs}, for point {j} ")
            gts.append(np.array(gt))
        return inputs, gts


def KMsubsetFromIndices(arr, indices):
    return [arr[i] for i in indices]


def KMconcat(a, b):
    out = []
    for v in a:
        out.append(v)
    for v in b:
        out.append(v)
    return out


def KIN_MUS_sessions_get(path):
    print(f"Parsing file [{path}].")
    data = loadmat(path)["EMG_KIN_v4"][0]
    print(f"Parsing {len(data)} lines of data.")
    sessions = []
    for s in data:
        session = KMSession(s[0], s[1], s[2], s[3], s[4], s[5])
        if session.ok:
            sessions.append(session)
    print(f"Found {len(sessions)} valid sessions.")
    return sessions


def KMSessionsAutoPrepare(KMsessions, inputLen, gtLen):
    inputs, gts = [], []
    for s in KMsessions:
        # Extract session data
        i, g = s.slice_sequence_into_InputsGts(inputLen, gtLen)
        inputs = KMconcat(inputs, i)
        gts = KMconcat(gts, g)
    return inputs, gts


def KMSession2InputsGts(session, inputLen, gtLen, verbose=False):
    inputs, gts = [], []
    # Extract session data
    i, g = session.slice_sequence_into_InputsGts(inputLen, gtLen)

    if verbose:
        i0 = i[0]
        g0 = g[0]
        print("Input:")
        print(f"type {type(i0)}, dims [{len(i0)}, {len(i0[0])}]")
        print(i0)
        print("Ground truth:")
        print(f"type {type(g0)}, dims [{len(g0)}, {len(g0[0])}]")
        print(g0)
        verbose = False
    inputs = KMconcat(inputs, i)
    gts = KMconcat(gts, g)
    return inputs, gts


def KMSessions2InputsGts(sessions, n_sessions, inputLen, gtLen):
    example = True
    if n_sessions < len(sessions) and n_sessions > 0:
        sessions = sessions[:n_sessions]

    inputs, gts = [], []
    for s in sessions:
        # Extract session data
        i, g = s.slice_sequence_into_InputsGts(inputLen, gtLen)

        if example:
            i0 = i[0]
            g0 = g[0]
            print("Input:")
            print(f"type {type(i0)}, dims [{len(i0)}, {len(i0[0])}]")
            print(i0)
            print("Ground truth:")
            print(f"type {type(g0)}, dims [{len(g0)}, {len(g0[0])}]")
            print(g0)
            example = False
        inputs = KMconcat(inputs, i)
        gts = KMconcat(gts, g)
    # print(f"input dataset len = {len(inputs)}")
    # print(f"gt dataset len={len(gts)}")
    return inputs, gts


if __name__ == "__main__":
    sessions = KIN_MUS_sessions_get('datasets/KIN_MUS_UJI.mat')
    for i in range(3):
        print(f"{i}) -> {sessions[i].as_str()}")
