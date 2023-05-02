import csv

class Marker():
    """
    Marker transformation container
    """
    def __init__(self):
        """
        pos: position specified in euclidean coordinates (xyz)
        quat: rotation specified in quaternion angles, (WXYZ)
        """
        self.pos = [0,0,0]
        self.rot = [1,0,0,0]
        self.has_rot = False


    def set_pos(self, x,y,z):
        self.pos = [x,y,z]


    def set_rot(self, w,x,y,z):
        self.rot = [w,x,y,z]
        self.has_rot = True


    def __str__(self):
        out = f"pos {self.pos[0]}, {self.pos[1]}, {self.pos[2]}"
        if self.has_rot:
            out += f"rot {self.rot[0]}, {self.rot[1]}, {self.rot[2]}, {self.rot[3]}"
        return  out
#markerFrame

class motiveParser():
    """
    CSV parser for Marker transformations.
    NOTE: Currently only expects non-rigidbody markers, eg. NO rotation information should exist.
    """
    def __init__(self, _csvpath):
        self.datarows = []
        with open(_csvpath, 'r') as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                self.datarows.append(row)
        self._interpretHeader()
        self._countMarkers()
        self._getFrameTimes()
        self._createMarkerMatrix()
        self._interpretPositions()

    def _interpretHeader(self):
        """
        Extract needed header information from parsed csv.
        Extracts:
        - take name
        - Date
        - Capture frameRate
        - Capture length
        - unit length
        """
        h = self.datarows[0]
        self.takename   = h[h.index('Take Name') + 1]
        self.datetime   = h[h.index('Capture Start Time') + 1]
        self.framerate  = h[h.index('Export Frame Rate') + 1]
        self.framecount = h[h.index('Total Exported Frames') + 1]
        self.unittype   = h[h.index('Length Units') + 1]


    def _countMarkers(self):
        """
        get amount of markers for each frame
        """
        self.markercount = self.datarows[6].count('X')
        return self.markercount


    def _getFrameTimes(self):
        """
        get array of frametimes of each marker recording
        """
        data_start = 7
        self.frametimes = []
        for i in range(0, int(self.framecount)-1):
            self.frametimes.append(self.datarows[i+data_start][1])
        return self.frametimes


    def _createMarkerMatrix(self):
        """
        Construct a matrix of markers of size:
        rows = framcecount
        columns = markercount
        """
        self.markers = []
        for _ in range(0, int(self.framecount)):
            markerframes = []
            for _ in range(0, self._countMarkers()):
                markerframes.append(Marker())
            self.markers.append(markerframes)


    def _interpretPositions(self):
        """
        Extract pos,rot information for each marker

        Structure of data:
        Frame, Time, pos1, ..., posN
        """
        data_start = 7

        for i in range(0, int(self.framecount) - 1):
            for j in range(0, self.markercount - 1):
                # easier formatting
                r = data_start + i
                c = 2 + (j*3)
                #print(f"marker index ({i},{j}) becomes csv  index ({r},{c})")
                if self.datarows[r][c] != '':
                    pos_x = float(self.datarows[r][c])
                else:
                    pos_x = 0.0

                if self.datarows[r][c + 1] != '':
                    pos_y = float(self.datarows[r][c + 1])
                else:
                    pos_y = 0.0

                if self.datarows[r][c + 2] != '':
                    pos_z = float(self.datarows[r][c + 2])
                else:
                    pos_z = 0.0

                self.markers[i][j].set_pos(pos_x, pos_y, pos_z)


    def printCsvFile(self):
        """
        print entire csv file
        """
        print(self.datarows)

    def getHeaderString(self):
        """
        return string of relevant header information for recording
        """
        # TODO: What unit type is framerate in? ms? hz?
        out =       f"{self.takename} : {self.datetime}\n"
        out = out + f"Frame Count:   {self.framecount}\n"
        out = out + f"Framerate:     {self.framerate}\n"
        out = out + f"Unit Type:     {self.unittype}\n"
        out = out + f"Marker Count:  {self.markercount}\n"
        return out


    def printHeader(self):
        """
        print relevant header information for recording
        """
        print(self.getHeaderString())


    def printMarker(self, _markerIndex):
        if _markerIndex < 0 or _markerIndex > self.markercount:
            print(f"Invalid marker (_markerIndex)")
            return
        for i in range(0, int(self.framecount)-1):
            print(f"Marker ({_markerIndex} : {self.frametimes[i]}) -> {self.markers[i][_markerIndex]}")
            #print(f"Marker ({_markerIndex} : {self.frametimes[i]}):", end='\t')
            #print(f"{self.markers[i][_markerIndex].pos[0]}, ", end = ' ')
            #print(f"{self.markers[i][_markerIndex].pos[1]}, ", end = ' ')
            #print(f"{self.markers[i][_markerIndex].pos[2]}, ", end = ' ')
            #print("")
#class markerParser


if __name__ == "__main__":
    print("Running test:\n----")
    path = './tjens18_test_movement01_xyz_marker_track.csv'
    mp = motiveParser(path)
    mp.printHeader()
    mp.printMarker(5)
