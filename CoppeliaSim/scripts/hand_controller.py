#python

from std_msgs.msg import Float32

# ------------------------------------------------------------------------
# Joint Class that manages external ROS2 Communication
class Joint():
    
    def setPositionCallback(msg):
        self.setPosition(msg.data)

    def __init__(self, path, name):
        msg_type = 'std_msgs/msg/Float32'
        self.is_invalid = 0
        self.handle = -1
        self.name = name
        self.path = path
        self.publisher_name = "/"+self.name+"_Get"
        self.subscriber_name = "/"+self.name+"_Set"

        # Find Joint Handle and validate it
        self.handle = sim.getObject(path)
        if not sim.isHandle(self.handle):
            sim.addLog(sim.verbosity_scriptinfos, "ERROR ["+self.path+"] Cant be found!")
            self.is_invalid = 1
            return self

        # Initialize Joint position
        self.setPosition(0)
        log = ""
        # Setup Publisher for external joint getter
        self.publisher = simROS2.createPublisher(self.publisher_name, msg_type)
        log += "Created publisher "+self.publisher_name+" ["+msg_type+"]"

        # Setup Subscriber for external joint setter
        self.subscriber = simROS2.createSubscription(self.subscriber_name,
                                                     msg_type,
                                                     self.setPositionCallback)
        log += "Created subscriber "+self.subscriber_name+" ["+msg_type+"]"
        
        if log == "":
            sim.addLog(sim.verbosity_scriptinfos, "Could not create Pub/Sub")
        else:
            sim.addLog(sim.verbosity_scriptinfos, log)

        return self

    def setPosition(self, pos):
        log = f"setting position for [{self.name}] to {pos}"
        sim.addLog(sim.verbosity_scriptinfos, log)
        sim.setJointTargetPosition(self.handle, pos)

    def getPath(self):
        return self.path

    def run(self):
        # Check if everything was setup correctly
        if self.is_invalid == 1:
            sim.addLog(sim.verbosity_scriptinfos, "ERROR ["+self.name+"] Cant be published!")
            return
        # Publish Joint Position
        msg = Float32()
        #msg.data = sim.getJointPosition(self.handle)
        print(f"HANDLE = {self.handle}")
        msg.data = sim.getJointPosition(self.handle)
        simROS2.publish(self.publisher, msg)

    def delete(self):
        # sim.addLog(sim.verbosity_scriptinfos, "Deleting ["+self.name+"]")
        if simROS2 and not self.is_invalid:
            simROS.shutdownPublisher(self.publisher)
            simROS.shutdownSubscriber(self.subscriber)
        return 1

# ------------------------------------------------------------------------
# Finger Class


def FingerCreate(path, name):
    rosname = name+"0"
    currpath = path+"/"+rosname+"/"+rosname+"_Joint"
    sim.addLog(sim.verbosity_scriptinfos, "Currpath ["+currpath+"]")
    Joint0 = Joint(currpath, rosname)

    rosname = name+"1"
    currpath = currpath+"/"+rosname+"/"+rosname+"_Joint"
    sim.addLog(sim.verbosity_scriptinfos, "Currpath ["+currpath+"]")
    Joint1 = Joint(currpath, rosname)

    rosname = name+"2"
    currpath = currpath+"/"+rosname+"/"+rosname+"_Joint"
    sim.addLog(sim.verbosity_scriptinfos, "Currpath ["+currpath+"]")
    Joint2 = Joint(currpath, rosname)
    return [Joint0, Joint1, Joint2]


def concat(a, b):
    out = []
    for v in a:
        out.append(v)
    for v in b:
        out.append(v)
    return out

# ---------------------------------------------------------------------------
# Main Syscalls
# ---------------------------------------------------------------------------


def sysCall_init():
    global Joints
    base_path = "/Wrist/Wrist_Joint/Finger_Contact/"

    Joints = []
    Joints.append(Joint("/Wrist/Wrist_Joint", "Wrist"))
    Joints = concat(Joints, FingerCreate(base_path+"Wrist_Index_Joint", "Index"))
    Joints = concat(Joints, FingerCreate(base_path+"Wrist_Index_Joint", "Thu"))
    Joints = concat(Joints, FingerCreate(base_path+"Wrist_Index_Joint", "Mid"))
    Joints = concat(Joints, FingerCreate(base_path+"Wrist_Index_Joint", "Rin"))
    Joints = concat(Joints, FingerCreate(base_path+"Wrist_Index_Joint", "Pin"))


def sysCall_actuation():
    global Joints
    for j in Joints:
        j.run()


def sysCall_cleanup():
    global Joints
    for j in Joints:
        j.delete()
