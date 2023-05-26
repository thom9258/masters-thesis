#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32


class TH_Joint(Node):
    def __init__(self, jointname):
        super().__init__(jointname)
        self.jointname = jointname
        self.topic = f"/{self.jointname}_Set"
        self.velocity_publisher = self.create_publisher(Float32, self.topic, 10)
        self.get_logger().info(f"Created Publisher [{self.topic}]")

    def sendPosition(self, pos):
        msg = Float32()
        msg.data = float(pos)
        self.get_logger().info(f"Published to [{self.topic}] with {msg.data}")
        # Publish 3 times to make sure rotation goes through
        for _ in range(0, 3):
            self.velocity_publisher.publish(msg)


class TH_Finger():
    def __init__(self, name):
        self.bend_joint = TH_Joint(name+"_Rotator")
        self.joints = []
        for i in range(0, 3):
            self.joints.append(TH_Joint(name+str(i)+"_Joint"))

    def spin(self):
        for j in self.joints:
            rclpy.spin(j)

    def flex(self, p0, p1, p2):
        self.joints[0].sendPosition(p0)
        self.joints[1].sendPosition(p1)
        self.joints[2].sendPosition(p2)

    def bend(self, p):
        self.bend_joint.sendPosition(p)


class TH_Hand():
    def __init__(self):
        self.wrist_bend = TH_Joint("Wrist_Rotator")
        self.thumb_rotator = TH_Joint("Thu_Rotator")
        self.wrist_flex = TH_Joint("Wrist_Flexor")
        self.wrist_index = TH_Joint("Wrist_Index_Joint")
        self.wrist_middle = TH_Joint("Wrist_Middle_Joint")
        self.wrist_ring = TH_Joint("Wrist_Ring_Joint")
        self.wrist_pinky = TH_Joint("Wrist_Pinky_Joint")

        self.finger_thumb = TH_Finger("Thu")
        self.finger_index = TH_Finger("Index")
        self.finger_middle = TH_Finger("Mid")
        self.finger_ring = TH_Finger("Rin")
        self.finger_pinky = TH_Finger("Pin")

    def wristBend(self, pos):
        self.wrist_bend.sendPosition(pos)

    def wristFlex(self, pos):
        flex = pos/2
        self.wrist_flex.sendPosition(flex)
        self.wrist_index.sendPosition(flex)
        self.wrist_middle.sendPosition(flex)
        self.wrist_ring.sendPosition(flex)
        self.wrist_pinky.sendPosition(flex)

    def spin(self):
        rclpy.spin(self.wrist_bend)
        rclpy.spin(self.wrist_flex)
        rclpy.spin(self.wrist_thumb)
        rclpy.spin(self.wrist_index)
        rclpy.spin(self.wrist_middle)
        rclpy.spin(self.wrist_ring)
        rclpy.spin(self.wrist_pinky)
        self.finger_thumb.spin()
        self.finger_index.spin()
        self.finger_middle.spin()
        self.finger_ring.spin()
        self.finger_pinky.spin()


def main(args=None):
    print("LIB NOT MEANT TO BE CALLED AS NODE!")


if __name__ == "__main__":
    main()
