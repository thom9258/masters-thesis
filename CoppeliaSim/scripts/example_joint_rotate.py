#python

def sysCall_init():
    global JointName
    global JointHandle

    JointName = "Joint"
    JointHandle = sim.getObjectHandle(JointName)
    pass

def sysCall_actuation():
    global JointName
    global JointHandle
    
    print(f"joint handle for [{JointName}] = {JointHandle}")
    sim.setJointTargetPosition(JointHandle, 3.14/2)
    t = sim.getJointTargetPosition(JointHandle)
    print(f"target = {t}")
    pass
