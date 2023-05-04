#!/usr/bin/env python3

import sys
import csv

class motiveEditor():
    def __init__(self, _csvpath):
        # Read entire file
        self.csvpath = _csvpath
        self.datarows = []
        with open(_csvpath, 'r') as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                self.datarows.append(row)

    def __del__(self):
        self.write()

    def framelen(self):
        h = self.datarows[0]
        return int(h[h.index('Total Exported Frames') + 1])

    def markercount(self):
        return self.datarows[6].count('X')
                
    def write(self):
        # write file
        with open(self.csvpath, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            for row in self.datarows:
                writer.writerow(row)

    def writeSubset(self, filename, start, end):
        # Manipulate framecount
        n = end - start
        newheader = []
        for col in self.datarows[0]:
            newheader.append(col)
        newheader[newheader.index('Total Exported Frames') + 1] = f"{n-1}"
        print(f"Writing subset {start} -> {end} to {filename}")
        # Write subset
        with open(filename, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(newheader)
            for row in self.datarows[1:7]:
                writer.writerow(row)
            for i in range(start, end):
                writer.writerow(self.datarows[i + 7])

    def removeMarker(self, n):
            print(f"Removing marker {n} ({self.markercount()-1} left)")
            mofs = 2 + n*3
            # print(f"removing csv index = {mofs}->{mofs+3}")
            for i in range(5, len(self.datarows)):
                del self.datarows[i][mofs:mofs+3]
            

    def combineMarkers(self, rowA, rowB, force=False):
            print(f"Combining marker {rowA} with {rowB}")
            aofs = 2 + rowA*3
            bofs = 2 + rowB*3
            # print(f"moving csv index {bofs}->{bofs+3} to {aofs}->{aofs+3}")
            for i in range(6, len(self.datarows)):
                if self.datarows[i][bofs:bofs+3] != ["", "", ""]:
                    self.datarows[i][aofs:aofs+3] = self.datarows[i][bofs:bofs+3]
            self.removeMarker(rowB)
# motiveEditor

def main():
    target = "target.csv"
    print(f"Target: {target}")
    parser = motiveEditor(target)
    framelen = parser.framelen()
    markercount = parser.markercount()
    print(f"Loaded target: {target} with {framelen} frames & {markercount} markers.")

    if len(sys.argv) < 1:
        print("ERROR! Expected command \"remove X\" or \"combine X Y\"")
        return 1

    if sys.argv[1] == "remove":
        assert len(sys.argv) == 3
        X = int(sys.argv[2])
        assert X >= 0 
        assert X < markercount 
        parser.removeMarker(X)

    elif sys.argv[1] == "combine":
        assert len(sys.argv) == 4
        X = int(sys.argv[2])
        Y = int(sys.argv[3])
        assert X >= 0
        assert Y >= 0
        parser.combineMarkers(rowA=X, rowB=Y)

    elif sys.argv[1] == "subset":
        assert len(sys.argv) == 5
        s = int(sys.argv[2])
        e = int(sys.argv[3])
        Name = sys.argv[4]
        assert s >= 0
        assert e >= 0
        assert s < framelen
        assert e < framelen
        parser.writeSubset(filename=Name, start=s, end=e)

if __name__ == "__main__":
    main()
