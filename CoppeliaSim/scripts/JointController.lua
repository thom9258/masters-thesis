-- ------------------------------------------------------------------------
-- Joint Class that manages external ROS2 Communication
Joint = {
    is_invalid,
    name,
    handle,
    -- Publisher messages the current joint position to external controller
    publisher_name,
    publisher,
    -- Subscriber retrieves new joint positions the joint needs to travel to
    subscriber_name,
    subscriber,
    callback,
}

function anySetPositionCallback(msg)
    sim.addLog(sim.verbosity_scriptinfos, "Callback Called! with "..msg.data)
    --msg.data
    --joint:setPosition(pos)
end

function Joint:create(path)
    local msg_type = 'std_msgs/msg/Float32'
    self.is_invalid = 0
    self.path = path
    self.publisher_name = self.path.."_Get"
    self.subscriber_name = self.path.."_Set"
    self.callback = callback
    
    -- Find Joint Handle and validate it
    self.handle = sim.getObject(path)
    if not sim.isHandle(self.handle) then
        sim.addLog(sim.verbosity_scriptinfos, "ERROR ["..self.path.."] Cant be found!")
        self.is_invalid = 1
        return self
    end

    -- Initialize Joint position
    self:setPosition(0)
    
    function anySetPositionCallback(msg)
        local log = "Callback Called on ["..self.path.."] with "..msg.data
        sim.addLog(sim.verbosity_scriptinfos, log)
        self:setPosition(msg.data)
    end
    
    local log
    if simROS2 then
        -- Setup Publisher for external joint getter
        self.publisher = simROS2.createPublisher(self.publisher_name, msg_type)
        log = "Created publisher "..self.publisher_name.." ["..msg_type.."]"
        sim.addLog(sim.verbosity_scriptinfos, log)

        -- Setup Subscriber for external joint setter
        self.subscriber = simROS2.createSubscription(self.subscriber_name,
                                                     msg_type,
                                                     'setPositionCallback')
        log = "Created subscriber "..self.subscriber_name.." ["..msg_type.."]"
        sim.addLog(sim.verbosity_scriptinfos, log)
    end
    return self
end

function Joint:setPosition(pos)
    local log = "setting position for ["..self.path.."] to "..pos
    sim.addLog(sim.verbosity_scriptinfos, log)
    sim.setJointTargetPosition(self.handle, pos)
end

function Joint:getPath()
    return self.path
end

function Joint:run()
    -- Check if everything was setup correctly
    if self.is_invalid == 1 then
        sim.addLog(sim.verbosity_scriptinfos, "ERROR ["..self.path.."] Cant be published!")
        return
    end
    
    -- Publish Joint Position
    simROS2.publish(self.publisher, {data=sim.getJointPosition(self.handle)})
end

function Joint:delete()
    sim.addLog(sim.verbosity_scriptinfos, "Deleting ["..self.path.."]")
    if simROS2 and not self.is_invalid then
        simROS.shutdownPublisher(self.publisher)
        simROS.shutdownSubscriber(self.subscriber)
    end
    return 1
end

-- ---------------------------------------------------------------------------
-- Main Syscalls
-- ---------------------------------------------------------------------------

function setPositionCallback(msg)
    joint:setPosition(msg.data)
end

function sysCall_init()
    path = "/Wrist_Joint"
    joint = Joint:create(path)
end

function sysCall_actuation()
    joint:run()
end

function sysCall_cleanup()
    joint:delete()
end
