#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from th_hand_controller_lib import TH_Joint, TH_Finger, TH_Hand


def main(args=None):
    rclpy.init(args=args)
    hand = TH_Hand()

    # FIVE FINGER PINCH
    while 1:
        time.sleep(0.5)
        hand.wristBend(0.0)
        hand.wristFlex(0.2)
        hand.finger_thumb.bend(-3)
        hand.finger_thumb.flex( 1.6, 0.2, 0.2)
        hand.finger_index.flex( 1.4, 0.2, 0.2)
        hand.finger_middle.flex(1.4, 0.2, 0.7)
        hand.finger_ring.flex(  1.4, 0.2, 0.7)
        hand.finger_pinky.flex( 1.4, 0.2, 0.7)

    # hand.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
