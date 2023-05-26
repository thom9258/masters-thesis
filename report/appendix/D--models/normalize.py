def gtsFindMinMax(gts):
    maxval = -99990
    minval =  99999
    for i, gt in enumerate(gts):
        for val in gt[0]:
            if val > maxval:
                maxval = val
            if val < minval:
                minval = val 
    print(f"min = ({minval}), max = {maxval}")
    return minval, maxval

def gtsMinMaxNormalize(gts, minval, maxval):
    for i in range(len(gts)):
        # print(f"gtset      = {gts[i][0]}")
        norms = []
        for val in gts[i][0]:
            norms.append((val-minval)/(maxval-minval))
        gts[i][0] = norms
        # print(f"normalized = {gts[i][0]}")
    return gts


def sequenceUnNormalize(s, minval, maxval):
    for i, val in enumerate(s):
        s[i] = val * (maxval - minval) + minval
    return s
