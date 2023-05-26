import math
import numpy as np

def concat(a, b):
     out = []
     for v in a:
         out.append(v)
     for v in b:
         out.append(v)
     return out

def slice_session(session, length):
    inputs = []
    maxlen = len(session.time)
    # Slice up session into prediction sets
    for i in range(maxlen - length):
        inp = session.muscles[i:i+length]
        inputs.append(np.array(inp))
        # print(f"extracted ({i}) = {inp}")
    return inputs

def zerocrossing_extract(emgs):
    tresh = 0.1
    emgzc = []
    for data in emgs:
        zc = []
        for i in range(len(data)-1):
            put = 0
            if data[i] <= tresh and data[i+1] > tresh:
                put = 1
            if data[i] > tresh and data[i+1] <= tresh:
                put = 1
            zc.append(put)
        emgzc.append(zc)
        # emgzc = concat(emgzc, zc)
    return emgzc

def rms_extract(emgs):
    emgrms = []
    for data in emgs:
        n = len(data)
        sqrsum = 0
        for v in data:
            sqrsum += v*v
        rms = math.sqrt((1/n) * sqrsum)
        emgrms.append([rms])
    return emgrms

def features_extract(emgs):
    feature_inputs = []
    #print(f"input: {emgs[0]}")
    for inp in emgs: 
        # print(f"inp: {inp}")
        zc = zerocrossing_extract(inp)
        # print(f"zc: {zc}")
        rms = rms_extract(inp)
        # print(f"rms: {rms}")
        feature_input = []
        for i in range(len(inp)):
            feature_input = concat(feature_input, zc[i])
            feature_input = concat(feature_input, rms[i])
        feature_inputs.append([feature_input])
    return feature_inputs

class_reaching = 1
class_manipulation = 2
class_releasing = 3

def class2onehot(v):
    if v == class_reaching:
        return [1, 0, 0]
    if v == class_manipulation:
        return [0, 1, 0]
    if v == class_releasing:
        return [0, 0, 1]
    return [0, 0, 0]

def gts2onehot(gts):
     return [class2onehot(gt) for gt in gts]
 

def KMSessions2ClassifierInputsGts(sessions, inputlen):
    inputs = []
    gts = []
    for session in sessions:
        sliced = slice_session(session, inputlen)
        inputs = concat(inputs, sliced)
        gt = [session.phase for _ in range(len(sliced))]
        gts = concat(gts, gt)
    return inputs, gts
