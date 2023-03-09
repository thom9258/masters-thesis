#!/bin/bash

SRC="$HOME/ros2_colcon_ws/src"
DST="./ros-packages/"
NAME="ros_backup.zip"

zip -r "$DST$(date +'%F-%R')--$NAME" $SRC
