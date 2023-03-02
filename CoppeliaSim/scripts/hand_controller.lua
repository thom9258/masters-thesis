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
}

function Joint:create(path, name)
    local msg_type = 'std_msgs/msg/Float32'
    self.is_invalid = 0
    self.name = name
    self.path = path
    self.publisher_name = "/"..self.name.."_Get"
    self.subscriber_name = "/"..self.name.."_Set"
    
    -- Find Joint Handle and validate it
    self.handle = sim.getObject(path)
    if not sim.isHandle(self.handle) then
        sim.addLog(sim.verbosity_scriptinfos, "ERROR ["..self.path.."] Cant be found!")
        self.is_invalid = 1
        return self
    end

    -- Initialize Joint position
    self:setPosition(sim.getJointPosition(self.handle) + 0.4)
    
    local log
    if simROS2 then
        -- Setup Publisher for external joint getter
        self.publisher = simROS2.createPublisher(self.publisher_name, msg_type)
        log = "Created publisher "..self.publisher_name.." ["..msg_type.."]"
        sim.addLog(sim.verbosity_scriptinfos, log)

        -- Setup Subscriber for external joint setter
        self.subscriber = simROS2.createSubscription(self.subscriber_name,
                                                     msg_type, 'this:setPositionCallback')
        log = "Created subscriber "..self.subscriber_name.." ["..msg_type.."]"
        sim.addLog(sim.verbosity_scriptinfos, log)
    end
    return self
end

function Joint:getPath()
    return self.path
end

function Joint:setPositionCallback(msg)
    self:setPosition(msg.data)
end

function Joint:setPosition(pos)
    local log = "setting position for ["..self.name.."] to "..pos
    sim.addLog(sim.verbosity_scriptinfos, log)
    sim.setJointTargetPosition(self.handle, pos)
end

function Joint:run()
    -- Check if everything was setup correctly
    if self.is_invalid == 1 then
        sim.addLog(sim.verbosity_scriptinfos, "ERROR ["..self.name.."] Cant be published!")
        return
    end
    
    -- Publish Joint Position
    simROS2.publish(self.publisher, {data=sim.getJointPosition(self.handle)})
end

function Joint:delete()
    sim.addLog(sim.verbosity_scriptinfos, "Deleting ["..self.name.."]")
    if simROS2 and not self.is_invalid then
        simROS.shutdownPublisher(self.publisher)
        simROS.shutdownSubscriber(self.subscriber)
    end
    return 1
end

-- ------------------------------------------------------------------------
-- Finger Class
Finger = {
    Joint0,
    Joint1,
    Joint2,
}

function Finger:create(path, name)
    local rosname
    local currpath
    
    rosname = name.."0"
    currpath = path.."/"..rosname.."/"..rosname.."_Joint"
    self.Joint0 = Joint:create(currpath, rosname)
    --sim.addLog(sim.verbosity_scriptinfos, "Currpath ["..currpath.."]")
    
    rosname = name.."1"
    currpath = currpath.."/"..rosname.."/"..rosname.."_Joint"
    self.Joint1 = Joint:create(currpath, rosname)
    --sim.addLog(sim.verbosity_scriptinfos, "Currpath ["..currpath.."]")    

    rosname = name.."2"
    currpath = currpath.."/"..rosname.."/"..rosname.."_Joint"
    self.Joint2 = Joint:create(currpath, rosname)
    --sim.addLog(sim.verbosity_scriptinfos, "Currpath ["..currpath.."]") 
    
    return self
end

function Finger:run()
    self.Joint0:run()
    self.Joint1:run()
    self.Joint2:run()
end

function Joint:delete()
    self.Joint0:delete()
    self.Joint1:delete()
    self.Joint2:delete()
end

-- ---------------------------------------------------------------------------
-- ---------------------------------------------------------------------------
-- ---------------------------------------------------------------------------

function sysCall_init()
    local base_path = "/Wrist/Wrist_Joint/Finger_Contact/"
    index = Finger:create(base_path.."Wrist_Index_Joint", "Index")
    --thumb = Finger:create(base_path.."Wrist_Index_Joint", "Thu")
end

function sysCall_actuation()
    index:run()
    --thumb:run()
end

function sysCall_cleanup()
    index:delete()
    --thumb:delete()
end
