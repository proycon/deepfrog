

import sys
import os
import argparse
import glob
import logging
import random
import shutil
import yaml

from deepfrog.tagger import Tagger


if 'DEEPFROG_CONFDIR' in os.environ:
    DEFAULTCONFDIR = os.environ['DEEPFROG_CONFDIR']
else:
    DEFAULTCONFDIR = os.path.join(os.path.expanduser("~"), ".deepfrog")

class DeepFrog:
    def __init__(self, **kwargs):
        if 'conf' not in kwargs:
            raise Exception("No configuration specified")

        self.confdir = kwargs['confdir'] if 'confdir' in kwargs else DEFAULTCONFDIR
        self.load_configuration(kwargs['conf'])

    def load_configuration(self, configfile):
        if configfile[:-4] != ".yml" or configfile[:-5] != ".yaml":
            configfile += ".yml"

        if os.path.exists(configfile):
            pass
        elif os.path.exists(os.path.join(self.confdir, configfile)):
            configfile = os.path.join(self.confdir, configfile)
        else:
            raise FileNotFoundError("Configuration " + configfile + " was not found in the search path. Set --confdir or $DEEPFROG_CONFDIR")
        with open(configfile,'r',encoding='utf-8') as f:
            self.config = yaml.load(f, Loader = yaml.CLoader)

    @staticmethod
    def argument_parser(parser=None):

        if parser is None:
            parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(description="DeepFrog", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-d','--outputdir', type=str,help="Output directory", action='store',default=".",required=False)
        parser.add_argument('--confdir', type=str,help="Configuration directory: the location where deepfrog configuration and models can be found", action='store',default=DEFAULTCONFDIR,required=False)
        parser.add_argument('-f','--format', type=str,help="Output format. Can be set to tsv for tab-seperated (columned) output, xml for FoLiA XML. If not set, the default will be tsv for plain text input, xml for FoLiA XML input.", action='store',required=False)
        parser.add_argument('-c','--conf', type=str, help="DeepFrog configuration file")
        parser.add_argument('-s','--skip', type=str, help="Skip the specified modules")
        parser.add_argument('input file', nargs='+', help="Input file (plain text or FoLiA XML)")
        return parser

def main():
    args = DeepFrog.argument_parser().parse_args()
    deepfrog = DeepFrog(**args.__dict__)

if __name__ == '__main__':
    main()
