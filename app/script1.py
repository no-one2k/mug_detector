#!/usr/bin/python

import sys, getopt
import numpy as np
import pandas as pd
import csv
import os
import utils

helpline = """Usage:
script1.py -i <input video file> -d <detector type: blue or yolo> -o <output video file name>
default output file name is "out.mp4"
default detector type is "blue"

script1.py -h print this helpline"""

    
def main(argv):
    input_file = None
    det_type = utils.DEFAULT_DETECTOR
    result_file = 'out.gif'
    try:
        opts, args = getopt.getopt(argv,"hi:d:o:",["ifile=", "detector=", "ofile="])
    except getopt.GetoptError:
        print 
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpline)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-d", "--detector"):
            det_type = arg
        elif opt in ("-o", "--ofile"):
            result_file = arg
    print('Input file is "',input_file, '"')
    assert(os.path.exists(input_file))
    print('Detector type is "', det_type,'"')
    assert(det_type in [utils.YOLO_DETECTOR, utils.BLUE_DETECTOR])
    print('Output file is "',result_file,'"')
    utils.process_video(input_file, det_type, result_file)

if __name__ == "__main__":
    main(sys.argv[1:])
