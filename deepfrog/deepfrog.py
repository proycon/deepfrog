

import sys
import os
import argparse
import glob
import logging
import random
import shutil
import yaml

from deepfrog.tagger import Tagger
from deepfrog import VERSION


if 'DEEPFROG_CONFDIR' in os.environ:
    DEFAULTCONFDIR = os.environ['DEEPFROG_CONFDIR']
else:
    DEFAULTCONFDIR = os.path.join(os.path.expanduser("~"), ".deepfrog")


REQUIREDFIELDS = ('name','module','foliatype','foliaset')

class ConfigurationError(Exception):
    pass

class DeepFrog:
    def __init__(self, **kwargs):
        if 'conf' not in kwargs:
            raise ConfigurationError("No configuration specified")

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

        if 'requireversion' in self.config:
            #Check version
            req = str(self.config['requireversion']).split('.')
            ver = str(VERSION).split('.')

            uptodate = True
            for i in range(0,len(req)):
                if i < len(ver):
                    if int(req[i]) > int(ver[i]):
                        uptodate = False
                        break
                    elif int(ver[i]) > int(req[i]):
                        break
            if not uptodate:
                raise ConfigurationError("Version mismatch: DeepFrog " + str(self.config['requireversion']) + " is required")

        if 'configversion' not in self.config:
            raise ConfigurationError("Configuration does not define a configversion")

        if 'language' not in self.config:
            raise ConfigurationError("Configuration does not define a language")

        if 'modules' not in self.config or not isinstance(self.config['modules'], (list,tuple)):
            raise ConfigurationError("No modules defined in configuration")

        for module in self.config['modules']:
            for field in REQUIREDFIELDS:
                if field not in module:
                    raise ConfigurationError("Missing field for module: " + field)

            if 'parameters' in module:
                for key, value in module['parameters'].items():
                    if isinstance(value, str):
                        module['parameters'][key] = value.replace("$ROOT", self.confdir)

    @staticmethod
    def argument_parser(parser=None):

        if parser is None:
            parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(description="DeepFrog", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-d','--outputdir', type=str,help="Output directory", action='store',default=".",required=False)
        parser.add_argument('--confdir', type=str,help="Configuration directory: the location where deepfrog configuration and models can be found", action='store',default=DEFAULTCONFDIR,required=False)
        parser.add_argument('-f','--format', type=str,help="Output format. Can be set to tsv for tab-seperated (columned) output, xml for FoLiA XML. If not set, the default will be tsv for plain text input, xml for FoLiA XML input.", action='store',required=False)
        parser.add_argument('-c','--conf', type=str, help="DeepFrog configuration file")
        parser.add_argument('-s','--skip', type=str, help="Skip the specified modules, takes a comma separated list of module names (names differ per configuration)")
        parser.add_argument('input file', nargs='+', help="Input file (plain text or FoLiA XML)")
        return parser

def main():
    args = DeepFrog.argument_parser().parse_args()
    deepfrog = DeepFrog(**args.__dict__)

if __name__ == '__main__':
    main()
