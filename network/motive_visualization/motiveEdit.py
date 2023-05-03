#!/usr/bin/env python3

from motiveParser import *
import sys

def main():
    if len(sys.argv) < 1:
        print("ERROR! Expected command \"remove X\" or \"combine X Y\"")
        return 1

    if sys.argv[1] == "remove":
        assert len(sys.argv) == 3
        X = int(sys.argv[2])
        print(f"wants to remove {X}")
        assert X > 0

    elif sys.argv[1] == "combine":
        assert len(sys.argv) == 4
        X = int(sys.argv[2])
        Y = int(sys.argv[3])
        print(f"wants to combine {X} and {Y}")
        assert X > 0
        assert Y > 0

if __name__ == "__main__":
    main()
