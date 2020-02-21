#!/usr/bin/env python

import logging
import random
import argparse

from deepfrog.data import TaggerInputDataset

ASSIGN_TRAIN = 0
ASSIGN_DEV = 1
ASSIGN_TEST = 2


def main():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test','-t', type=int,help="Size of test set", action='store',default=100,required=True)
    parser.add_argument('--dev','-d', type=int,help="Size of dev set", action='store',default=0,required=True)
    parser.add_argument('--seed','-s', help="Seed for random number generator", type=int, action='store',default=0)
    parser.add_argument('input', nargs='+', help='input files')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    logger = logging.getLogger(__name__)

    data = TaggerInputDataset(logger)
    for inputfile in args.input:
        data.load_file(inputfile)

    assignments = [ASSIGN_TRAIN] * len(data)

    dev = set()
    test = set()

    while len(dev) < args.dev:
        index = random.randint(0, len(data))
        if assignments[index] == ASSIGN_TRAIN:
            assignments[index] = ASSIGN_DEV
            dev.add(index)

    while len(test) < args.test:
        index = random.randint(0, len(data))
        if assignments[index] == ASSIGN_TRAIN:
            assignments[index] = ASSIGN_TEST
            test.add(index)

    ftrain = open("train",'w',encoding='utf-8')
    ftest = open("test",'w',encoding='utf-8')
    fdev = open("dev",'w',encoding='utf-8')

    for i, assignment in enumerate(assignments):
        if assignment == ASSIGN_TRAIN:
            data[i].write(ftrain)
        elif assignment == ASSIGN_DEV:
            data[i].write(fdev)
        elif assignment == ASSIGN_TEST:
            data[i].write(ftest)



if __name__ == '__main__':
    main()








