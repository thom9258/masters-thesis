function subscriber_callback(msg)
    -- This is the subscriber callback function
    sim.addLog(sim.verbosity_scriptinfos,'subscriber receiver following Float32: '..msg.data)
end

function getTransformStamped(objHandle,name,relTo,relToName)
    -- This function retrieves the stamped transform for a specific object
    t=simROS2.getSystemTime()
    p=sim.getObjectPosition(objHandle,relTo)
    o=sim.getObjectQuaternion(objHandle,relTo)
    return {
        header={
            stamp=t,
            frame_id=relToName
        },
        child_frame_id=name,
        transform={
            translation={x=p[1],y=p[2],z=p[3]},
            rotation={x=o[1],y=o[2],z=o[3],w=o[4]}
        }
    }
end

function sysCall_init()
    -- The child script initialization
    objectHandle=sim.getObject('.')
    objectAlias=sim.getObjectAlias(objectHandle,3)

    -- Prepare the float32 publisher and subscriber (we subscribe to the topic we publish):
    if simROS2 then
        publisher=simROS2.createPublisher('/simulationTime','std_msgs/msg/Float32')
        subscriber=simROS2.createSubscription('/simulationTime','std_msgs/msg/Float32','subscriber_callback')
    end
end

function sysCall_actuation()
    -- Send an updated simulation time message, and send the transform of the object attached to this script:
    if simROS2 then
        simROS2.publish(publisher,{data=sim.getSimulationTime()})
        simROS2.sendTransform(getTransformStamped(objectHandle,objectAlias,-1,'world'))
        -- To send several transforms at once, use simROS2.sendTransforms instead
    end
end

function sysCall_cleanup()
    -- Following not really needed in a simulation script (i.e. automatically shut down at simulation end):
    if simROS2 then
        simROS.shutdownPublisher(publisher)
        simROS.shutdownSubscriber(subscriber)
    end
end
