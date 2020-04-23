

import sys
import os
import argparse
import glob
import logging
import random
import shutil
import yaml
import pickle

from deepfrog.tagger import Tagger
from deepfrog import VERSION


if 'DEEPFROG_CONFDIR' in os.environ:
    DEFAULTCONFDIR = os.environ['DEEPFROG_CONFDIR']
else:
    DEFAULTCONFDIR = os.path.join(os.path.expanduser("~"), ".deepfrog")


REQUIREDFIELDS = ('name','module','foliatype','foliaset')

MODULES = {
    "Tagger": Tagger,
}

logger = logging.getLogger(__name__)



class ConfigurationError(Exception):
    pass

class DeepFrog:
    def __init__(self, **kwargs):
        if 'conf' not in kwargs:
            raise ConfigurationError("No configuration specified")
        if 'logger' in kwargs:
            self.logger = kwargs['logger']
        else:
            self.logger = logging.getLogger(__name__)
        self.logger.info("Initialising DeepFrog")

        self.confdir = kwargs['confdir'] if 'confdir' in kwargs else DEFAULTCONFDIR
        self.logger.info("Configuration directory: %s", self.configdir)

        self.input_cache_dir = kwargs['input_cache_dir'] if 'input_cache_dir' in kwargs else '.'
        self.logger.info("Input cache dir: %s", self.input_cache_dir)

        self.load_configuration(kwargs['conf'])

        self.args = kwargs

        #Initialize
        self.logger.info("Loading modules")
        self.modules = []
        for module in self.config['modules']:
            self.logger.info("Loading module %s" , module['name'])
            if not kwargs['skip'] or (kwargs['skip'] and module['name'] not in kwargs['skip'].split(",")):
                ModuleClass = MODULES[module['module']]
                parameters = module['parameters']
                parameters['logger'] = self.logger
                self.modules.append(ModuleClass(**parameters))

    def load_configuration(self, configfile):
        self.logger.info("Loading configuration %s", configfile)

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
                        module['parameters'][key] = value.replace("$INPUT_CACHE_DIR", self.args['input_cache_dir'])

    def process(self, inputfile, outputfile, inputformat, outputformat, **kwargs):
        """Process an entire document"""
        output = {} #each module will write an output layer
        for module in self.modules:
            #Convert the input file to something the module can handle (moduleinputfile)
            moduleinputfile = os.path.join(self.input_cache_dir, os.path.basename(inputfile) + "." + module.InputFormat.extension)
            if os.path.exists(moduleinputfile): #is it cached already?
                with open(moduleinputfile + ".alignment",'rb') as f:
                    alignment = pickle.load(f)
            else:
                if hasattr(module.InputFormat,"from_" +  inputformat):
                    #create the intermediate input file from the original one and return the alignment
                    alignment = getattr(module.InputFormat,"from_" + inputformat)(inputfile, moduleinputfile, **kwargs)
                    #cache the alignment for next time
                    with open(moduleinputfile + ".alignment",'wb') as f:
                        pickle.dump(alignment, f)
                else:
                    raise Exception("Module " + str(module.__class__.__name__) + " can't handle input format " + str(inputformat))
            #call the module
            output[module.name] = module(test_file=moduleinputfile)
        return output

    @staticmethod
    def argument_parser(parser=None):

        if parser is None:
            parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(description="DeepFrog", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-d','--outputdir', type=str,help="Output directory", action='store',default=".",required=False)
        parser.add_argument('--confdir', type=str,help="Configuration directory: the location where deepfrog configuration and models can be found", action='store',default=DEFAULTCONFDIR,required=False)
        parser.add_argument('-f','--format', type=str,help="Output format. Can be set to tsv for tab-separated (columned) output, xml for FoLiA XML. If not set, the default will be tsv for plain text input, xml for FoLiA XML input.", action='store',required=False)
        parser.add_argument('--inputformat', type=str,help="Input format. Can be set to txt for plain text or xml for FoLiA XML", action='store',required=False)
        parser.add_argument('-c','--conf', type=str, help="DeepFrog configuration file")
        parser.add_argument('-s','--skip', type=str, help="Skip the specified modules, takes a comma separated list of module names (names differ per configuration)")
        parser.add_argument(
            "--input_cache_dir",
            default=".",
            type=str,
            help="Where do you want to store cached data for the input data? (defaults to current working directory)",
        )
        parser.add_argument('inputfiles', nargs='+', help="Input file (plain text or FoLiA XML)")
        return parser

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = DeepFrog.argument_parser().parse_args()
    deepfrog = DeepFrog(**args.__dict__)
    for inputfile in args.inputfiles:
        if inputfile.lower().endswith(".txt"):
            inputformat = "txt"
            inputbase = inputfile[:-4]
        elif inputfile.lower().endswith(".xml"):
            inputformat = "xml"
            inputbase = inputfile[:-4]
            if inputbase.lower().endswith(".folia"):
                inputbase = inputbase[:-6]
        elif args.inputformat:
            inputformat = args.inputformat
        else:
            raise Exception("Unable to derived input format from file extension, please set explicitly with --inputformat")

        if args.outputformat:
            outputformat = args.outputformat
        elif inputformat == "txt":
            outputformat = "tsv"
        elif inputformat == "xml":
            outputformat = "xml"

        if outputformat == "xml":
            outputfile = os.path.join(args.outputdir, inputbase + '.folia.xml')
        else:
            outputfile = os.path.join(args.outputdir, inputbase + '.' + outputformat)

        deepfrog.process(inputfile, outputfile, inputformat, outputformat)



if __name__ == '__main__':
    main()
