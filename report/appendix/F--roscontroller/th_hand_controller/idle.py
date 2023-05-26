#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from th_hand_controller_lib import TH_Joint, TH_Finger, TH_Hand


def main(args=None):
    rclpy.init(args=args)
    hand = TH_Hand()
    while 1:
        time.sleep(0.5)
        hand.wristBend(0.2)
        hand.wristFlex(0.4)
        hand.finger_thumb.flex(0.3, 0.3, 0.4)
        hand.finger_thumb.bend(-0.2)
        hand.finger_index.flex(0.3, 0.3, 0.4)
        hand.finger_middle.flex(0.3, 0.3, 0.4)
        hand.finger_ring.flex(0.3, 0.3, 0.4)
        hand.finger_pinky.flex(0.3, 0.3, 0.4)

    # hand.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
