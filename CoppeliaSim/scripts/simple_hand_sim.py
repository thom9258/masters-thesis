#python

def objPos(name, pos):
    h = sim.getObjectHandle(name)
    sim.setJointTargetPosition(h, pos)


def sysCall_init():
    global JointName
    global JointHandle

    JointName = "Wrist_Joint"
    JointHandle = sim.getObjectHandle(JointName)
    pass
    
def sysCall_actuation():
    global JointName
    global JointHandle
    
    print(f"joint handle for [{JointName}] = {JointHandle}")
    sim.setJointTargetPosition(JointHandle, 0.2)
    t = sim.getJointPosition(JointHandle)
    print(f"Pos = {t}")

    objPos("Wrist_Index_Joint", 0.2)
    objPos("Index0_Joint", 1)
    objPos("Index1_Joint", 1)
    objPos("Index2_Joint", 1)
    pass
