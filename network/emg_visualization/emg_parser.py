import csv
import matplotlib.pyplot as plt

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

def quickPlot(data, data_labels, axis_labels=["x", "y"]):
    plt.figure()
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    for i in range(0, len(data)):
        plt.plot(data[i], label=data_labels[i])
    plt.legend()
    plt.grid()
    plt.show()


def Buttersworth_Low(data, sampling_frequency, cutoff_frequency=10, order=5):
    # Nyquist Frequency Normalization
    w = cutoff_frequency / (sampling_frequency / 2) 
    b, a = signal.butter(order, w, btype='low')
    return signal.filtfilt(b, a, data)


def Buttersworth_Band(data, lowcut, highcut, sampling_frequency, order=5):
    # Nyquist Frequency Normalization
    wl = lowcut / (sampling_frequency / 2)
    wh = highcut / (sampling_frequency / 2)
    b, a = signal.butter(order, [wl, wh], fs=sampling_frequency, btype='band')
    return signal.filtfilt(b, a, data)


if __name__ == "__main__":
    print("TEST:")
    emgcsvfile = "../datasets/grasp_dataset/tjens18_index_30s_2_Trigno_2801.csv"
    parser = emgParser(emgcsvfile)
    tricep = parser.tricep[0::10]

    tricep_bwl = Buttersworth_Low(tricep,
                                  sampling_frequency=200,
                                  cutoff_frequency=10,
                                  order=5)

    tricep_bwb = Buttersworth_Band(tricep,
                                   sampling_frequency=200,
                                   lowcut=10,
                                   highcut=2000,
                                   order=5)
    
    quickPlot([tricep, tricep_bwl, tricep_bwb],
              ["tricep", "tricep Buttersworth Lowpass",  "tricep Buttersworth Bandpass"],
              axis_labels=["Time", "mV"])


