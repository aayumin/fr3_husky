#!/usr/bin/env python3

import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class KeyboardCmdPublisher(Node):
    def __init__(self):
        super().__init__("keyboard_cmd_publisher")
        self.pub = self.create_publisher(String, "/keyboard_cmd", 10)

    def publish_key(self, key: str):
        msg = String()
        msg.data = key
        self.pub.publish(msg)


def read_key():
    c = sys.stdin.read(1)

    if c == "\x1b":
        seq = sys.stdin.read(2)
        if seq == "[A":
            return "up"
        if seq == "[B":
            return "down"
        if seq == "[D":
            return "left"
        if seq == "[C":
            return "right"
        return "esc"

    return c


def main():
    rclpy.init()
    node = KeyboardCmdPublisher()

    old_attr = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        print("Keyboard publisher started. q: quit")
        while rclpy.ok():
            key = read_key()
            node.publish_key(key)
            if key == "q":
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()