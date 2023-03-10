from scipy.io import loadmat

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
                print(f"WARNING: angle/muscles is [{angles[0]}/{muscles}] and NOT [18/7]")
                return
        self.subject_id = subject_id
        self.adl = adl
        self.time = time
        self.phase = phase
        self.angles = angles
        self.muscles = muscles
        self.ok = True


def kin_mus_parse(path):
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


if __name__ == "__main__":
    sessions = kin_mus_parse('datasets/KIN_MUS_UJI.mat')
