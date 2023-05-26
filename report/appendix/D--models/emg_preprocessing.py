#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import itertools
from itertools import product
from th_ai import th_quickPlot

from scipy import signal 

class emgParser():
    def __init__(self, _csvpath):
        self.csvdata = []
        with open(_csvpath, 'r') as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                self.csvdata.append(row)
        self._interpretHeader()
        self.printHeader()
        self._interpretData()

    def _interpretHeader(self):
        self.takename   = self.csvdata[1][1]
        self.datetime   = self.csvdata[4][1]
        self.framerate  = self.csvdata[3][1]
        self.framecount = self.csvdata[5][1]
        self.dataheader = self.csvdata[14] 
        self.dataOffset = self.csvdata[15] 
        print("... Parsed Header data!")


    def header2str(self):
        out = ""
        out += f"takename   = {self.takename}\n"
        out += f"datetime   = {self.datetime}\n"
        out += f"framerate  = {self.framerate} Hz\n"
        out += f"framecount = {self.framecount}\n"
        out += "Data Header:\n"
        out += f"{self.dataheader}"
        return out

    def printHeader(self):
        print(self.header2str())

    def printCsvFile(self):
        """
        print entire csv file
        """
        print(self.csvdata)

    def _interpretData(self):

        def getCol(arr, colidx):
            col = []
            for row in arr:
                # float conversion because we only use this for emg data!
                col.append(float(row[colidx]))
            return col
            
        mocaptimeOffset = self.dataheader.index(" MocapTime")
        deviceframeOffset = self.dataheader.index(" DeviceFrame")
        pecOffset = self.dataheader.index("pec")
        arm1Offset = self.dataheader.index("arm1")
        arm2Offset = self.dataheader.index("arm2")
        arm3Offset = self.dataheader.index("arm3")
        bicepOffset = self.dataheader.index("bicep")
        tricepOffset = self.dataheader.index("tricep")

        data = self.csvdata[15:]
        self.deviceFrame = getCol(data, deviceframeOffset)
        self.pec = getCol(data, pecOffset)
        self.arm1 = getCol(data, arm1Offset)
        self.arm2 = getCol(data, arm2Offset)
        self.arm3 = getCol(data, arm3Offset)
        self.bicep = getCol(data, bicepOffset)
        self.tricep = getCol(data, tricepOffset)
        print("... Parsed sEMG data!")
#class emgParser

def uniquecombinations(arr1, arr2):
    res = []
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            res.append((arr1[i], arr2[j]))
    return res


def Notch(data, fs, fc, quality):
    # Nyquist Frequency Normalization
    w = fc / (fs / 2) 
    b, a = signal.iirnotch(w, quality, fs)
    return signal.filtfilt(b, a, data)


def Buttersworth_Low(data, sampling_frequency, cutoff_frequency, order):
    # Nyquist Frequency Normalization
    w = cutoff_frequency / (sampling_frequency / 2) 
    b, a = signal.butter(order, w, btype='low')
    return signal.filtfilt(b, a, data)


def Buttersworth_Band(data, lowcut, highcut, sampling_frequency, order):
    # Nyquist Frequency Normalization
    wl = lowcut / (sampling_frequency / 2)
    wh = highcut / (sampling_frequency / 2)
    b, a = signal.butter(order, [wl, wh], fs=sampling_frequency, btype='band')
    return signal.filtfilt(b, a, data)


def Buttersworth_lowpass_test(data, cutoffs, orders, title=None):
    fs = 200
    f_graphs = [data]
    f_labels = ["Raw"]
    combinations = uniquecombinations(cutoffs, orders)
    for pair in combinations:
        c = pair[0]
        o = pair[1]
        f_graphs.append(Buttersworth_Low(data, fs, c, o))
        f_labels.append(f"Buttersworth Lowpass (fc={c},order={o})")
    th_quickPlot(f_graphs, f_labels, axis_labels=["Time", "mV"], inches=[8, 4])
    return f_graphs, f_labels

def Buttersworth_bandpass_test(data, lowhighs, orders, title=None):
    fs = 200
    f_graphs = [data]
    f_labels = ["Raw"]
    combinations = uniquecombinations(lowhighs, orders)
    for pair in combinations:
        lh = pair[0]
        o = pair[1]
        f_graphs.append(Buttersworth_Band(data, lh[0], lh[1], fs, o))
        f_labels.append(f"Buttersworth Bandpass (lowhigh={lh},order={o})")
    th_quickPlot(f_graphs, f_labels, axis_labels=["Time", "mV"], inches=[8, 4])
    return f_graphs, f_labels

if __name__ == "__main__":
    print("TEST:")
    emgcsvfile = "./datasets/grasp_dataset/tjens18_index_30s_2_Trigno_2801.csv"
    parser = emgParser(emgcsvfile)
    emg = parser.arm1[0::10]
    emg = emg[:300]

    fs = 200

    bwf_20lp5or = Buttersworth_Low(data=emg,
                                   sampling_frequency=fs,
                                   cutoff_frequency=20,
                                   order=5)
    bwf_10lp10or = Buttersworth_Low(data=emg,
                                   sampling_frequency=fs,
                                   cutoff_frequency=10,
                                    order=10)


    bwf_10_500bp5or = Buttersworth_Band(data=emg,
                                        lowcut=10,
                                        highcut=500,
                                        sampling_frequency=fs,
                                        order=2)
    bwf_10_500bp5or_notch = Notch(data=bwf_10_500bp5or, fs=fs, fc=60, quality=6)


    bwf_20_800bp5or = Buttersworth_Band(data=emg,
                                        lowcut=20,
                                        highcut=800,
                                        sampling_frequency=fs,
                                        order=4)
    bwf_20_800bp5or_notch = Notch(data=bwf_20_800bp5or, fs=fs, fc=60, quality=6)




    th_quickPlot([emg,
                  bwf_20lp5or,
                  bwf_10lp10or,
                 ],
                 [
                     "Raw (Unfiltered)",
                     "Buttersworth Lowpass (20, 5)",
                     "Buttersworth Lowpass (10, 10)",
                 ],
                 axis_labels=["Samples", "mV"], inches=[8, 4])

    th_quickPlot([emg,
                  bwf_10_500bp5or,
                  bwf_10_500bp5or_notch,
                  # bwf_20_800bp5or_notch,
                 ],
                 [
                     "Raw (Unfiltered)",
                     "Buttersworth Bandpass ([10, 500], 2)",
                     "Buttersworth Bandpass ([10, 500], 2) + Notch (60)",
                   #  "Buttersworth Bandpass ([20, 1500], 2) + Notch (60)",
                 ],
                 axis_labels=["Samples", "mV"], inches=[8, 4])





    # Buttersworth tests
    if 0:
        Buttersworth_lowpass_test(emg,
                                  cutoffs=[10, 20, 30, 40, 50],
                                  orders=[10],
                                  title="Lowpass Cutoff Test")

        Buttersworth_lowpass_test(emg,
                                  cutoffs=[20],
                                  orders=[2, 5, 8, 10],
                                  title="Lowpass Order Test")

        Buttersworth_bandpass_test(emg,
                                lowhighs=[[10, 2000], [20, 2500], [50, 400]],
                                   orders=[5],
                                   title="Bandpass, Cutoff Test")

    # notch = Notch(tricep, fs=200, fc=100, quality=1)
    # quickPlot([tricep, notch], ["tricep", "notch"])



