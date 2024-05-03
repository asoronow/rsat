import numpy as np
import argparse

if __name__ == "__main__":
    '''
    Takes in a full size image and performs the RSAT segmentation algorithm.
    Tile by tile to perserver memory. Then it outputs the segmented image.

    Parameters:
        -i: The input image.
        -o: The output image.
        -s: The size of the tiles.
    '''

    parser = argparse.ArgumentParser(description="Segment a full size image.")
    parser.add_argument("-i", "--input", type=str, help="The input image.")
    parser.add_argument("-o", "--output", type=str, help="The output image.")
    parser.add_argument("-s", "--size", type=int, help="The size of the tiles.")
    args = parser.parse_args()

    if args.input is None or args.output is None or args.size is None:
        print("Invalid arguments.")
        exit(1)
    